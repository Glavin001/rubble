import type { ScenarioDesc } from "@/lib/stress/core/types";
import {
  type FragmentInfo,
  buildWallFragments,
  buildFloorFragments,
  buildColumnFragments,
  addFoundationFragments,
  buildScenarioFromFragments,
} from "./fractureUtils";

type FracturedTowerOptions = {
  /** Width of the tower footprint (X dimension) in meters */
  width?: number;
  /** Depth of the tower footprint (Z dimension) in meters */
  depth?: number;
  /** Number of floors (stories) in the building */
  floorCount?: number;
  /** Floor-to-floor height in meters (typical: 3.5-4m for commercial) */
  floorHeight?: number;
  /** Total height override - if provided, ignores floorCount × floorHeight */
  height?: number;
  /** Thickness of exterior walls in meters */
  thickness?: number;
  /** Thickness of floor slabs in meters */
  floorThickness?: number;
  /** Size of interior columns (square cross-section) in meters */
  columnSize?: number;
  /** Number of columns in X direction (grid). Default auto-calculated based on spacing. */
  columnsX?: number;
  /** Number of columns in Z direction (grid). Default auto-calculated based on spacing. */
  columnsZ?: number;
  /** Target spacing between columns in meters (used if columnsX/Z not specified). Default ~8m. */
  columnSpacing?: number;
  /** Inset from walls to first column row (fraction of interior span, 0-0.5). Default 0.15 */
  columnInset?: number;
  /** Number of fracture pieces per wall section (per floor) */
  fragmentCountPerWall?: number;
  /** Number of fracture pieces per floor plate */
  fragmentCountPerFloor?: number;
  /** Number of fracture pieces per column section */
  fragmentCountPerColumn?: number;
  /** Total mass of the structure in kg */
  deckMass?: number;
  /**
   * Use auto bonding instead of legacy bond computation for better performance.
   * When true, bonds array will be empty; apply applyAutoBondingToScenario after.
   */
  useAutoBonding?: boolean;
};

/**
 * Builds a multi-floor tower structure from fractured walls, floor plates, and interior columns.
 * Uses three-pinata fracturing for each component.
 * 
 * Realistic skyscraper defaults:
 * - 40m × 40m footprint (typical mid-rise office building)
 * - 4m floor-to-floor height (commercial standard)
 * - 20 floors (~80m total height)
 * - 4 exterior walls per floor section
 * - Interior columns in a configurable grid (default ~8m spacing)
 * - Floor plates between each story (full width, walls sit on top)
 * - Roof plate at top
 * 
 * Column grid options:
 * - columnsX/columnsZ: Specify exact grid dimensions
 * - columnSpacing: Auto-calculate grid based on target spacing (~8m default)
 * - columnInset: How far from walls to place first column row (0.15 = 15% inset)
 */
export async function buildFracturedTowerScenario({
  // Realistic skyscraper defaults (in meters)
  width = 40,
  depth = 40,
  floorCount = 20,
  floorHeight = 4,
  height,
  thickness = 0.4,
  floorThickness = 0.35,
  columnSize = 1.2,
  columnsX,
  columnsZ,
  columnSpacing = 8,
  columnInset = 0.15,
  fragmentCountPerWall = 12,
  fragmentCountPerFloor = 8,
  fragmentCountPerColumn = 4,
  deckMass,
  useAutoBonding = false,
}: FracturedTowerOptions = {}): Promise<ScenarioDesc> {
  // Calculate total height from floor count if not explicitly provided
  const totalHeight = height ?? floorCount * floorHeight;
  
  // Calculate realistic mass if not provided
  // ~600 kg/m² per floor (typical reinforced concrete office building)
  const floorArea = width * depth;
  const calculatedMass = deckMass ?? floorArea * floorCount * 600;

  // Foundation dimensions (scaled to building size)
  const foundationHeight = Math.max(0.5, totalHeight * 0.01);
  const groundClearance = Math.max(0.01, foundationHeight * 0.05);
  // Foundation top is at baseY - walls start directly on foundation
  const baseY = groundClearance + foundationHeight;

  // Calculate floor heights for each story
  // floorHeights[0] = baseY (ground floor - on foundation)
  // floorHeights[1] = baseY + floorHeight (first elevated floor)
  // etc.
  const floorHeights: number[] = [];
  for (let i = 0; i <= floorCount; i++) {
    floorHeights.push(baseY + i * floorHeight);
  }

  // Calculate wall positions (walls form the perimeter of the tower)
  const halfWidth = width * 0.5;
  const halfDepth = depth * 0.5;

  // Interior column grid calculation
  // Columns are placed within the interior (inset from walls)
  const interiorWidth = width - thickness * 2;
  const interiorDepth = depth - thickness * 2;
  
  // Calculate number of columns in each direction
  // If not specified, auto-calculate based on spacing
  const numColumnsX = columnsX ?? Math.max(2, Math.round(interiorWidth / columnSpacing));
  const numColumnsZ = columnsZ ?? Math.max(2, Math.round(interiorDepth / columnSpacing));
  
  // Calculate column positions in a grid
  // Inset from interior edges by columnInset fraction
  const columnRangeX = interiorWidth * (1 - 2 * columnInset);
  const columnRangeZ = interiorDepth * (1 - 2 * columnInset);
  const columnStartX = -columnRangeX * 0.5;
  const columnStartZ = -columnRangeZ * 0.5;
  
  const columnPositions: Array<{ x: number; z: number }> = [];
  for (let ix = 0; ix < numColumnsX; ix++) {
    for (let iz = 0; iz < numColumnsZ; iz++) {
      // Distribute columns evenly across the range
      const tx = numColumnsX > 1 ? ix / (numColumnsX - 1) : 0.5;
      const tz = numColumnsZ > 1 ? iz / (numColumnsZ - 1) : 0.5;
      columnPositions.push({
        x: columnStartX + tx * columnRangeX,
        z: columnStartZ + tz * columnRangeZ,
      });
    }
  }

  // All fragments from all walls, floors, and columns
  const allFragments: FragmentInfo[] = [];

  // Build walls for each floor section
  // Each wall section spans from one floor to the next
  for (let floorIdx = 0; floorIdx < floorHeights.length - 1; floorIdx += 1) {
    // Ground floor (idx 0): no floor plate, walls start directly at baseY
    // Upper floors: walls start at top of floor plate below
    const wallBottomY = floorIdx === 0
      ? floorHeights[0]  // Ground floor: start at foundation top
      : floorHeights[floorIdx] + floorThickness * 0.5;  // Upper floors: start at top of floor plate
    
    // Wall top is at bottom of floor plate above (or roof for top section)
    const wallTopY = floorHeights[floorIdx + 1] - floorThickness * 0.5;
    const wallHeight = wallTopY - wallBottomY;
    
    if (wallHeight <= 0.1) continue; // Skip if floor spacing is too small

    const wallLiftY = wallBottomY;

    // Front wall (at z = -halfDepth, spans X)
    const frontWallFrags = buildWallFragments(
      width,
      wallHeight,
      thickness,
      fragmentCountPerWall,
      0, // centerX
      -halfDepth + thickness * 0.5, // centerZ (inset by half thickness)
      0, // no rotation
      wallLiftY,
    );
    allFragments.push(...frontWallFrags);

    // Back wall (at z = +halfDepth, spans X)
    const backWallFrags = buildWallFragments(
      width,
      wallHeight,
      thickness,
      fragmentCountPerWall,
      0, // centerX
      halfDepth - thickness * 0.5, // centerZ (inset by half thickness)
      0, // no rotation
      wallLiftY,
    );
    allFragments.push(...backWallFrags);

    // Side wall span (fit between front/back walls)
    const sideWallSpan = depth - thickness * 2;

    // Left wall (at x = -halfWidth, spans Z)
    const leftWallFrags = buildWallFragments(
      sideWallSpan,
      wallHeight,
      thickness,
      fragmentCountPerWall,
      -halfWidth + thickness * 0.5, // centerX (inset by half thickness)
      0, // centerZ
      Math.PI * 0.5, // 90 degree rotation
      wallLiftY,
    );
    allFragments.push(...leftWallFrags);

    // Right wall (at x = +halfWidth, spans Z)
    const rightWallFrags = buildWallFragments(
      sideWallSpan,
      wallHeight,
      thickness,
      fragmentCountPerWall,
      halfWidth - thickness * 0.5, // centerX (inset by half thickness)
      0, // centerZ
      Math.PI * 0.5, // 90 degree rotation
      wallLiftY,
    );
    allFragments.push(...rightWallFrags);

    // Build interior columns for this floor section
    for (const colPos of columnPositions) {
      const columnFrags = buildColumnFragments(
        columnSize,
        columnSize,
        wallHeight,
        fragmentCountPerColumn,
        colPos.x,
        wallLiftY,
        colPos.z,
      );
      allFragments.push(...columnFrags);
    }
  }

  // Build floor plates (except ground floor which is handled by foundation)
  // Floor plates span the FULL width/depth so walls sit ON TOP of them
  // This ensures walls and floors have touching/overlapping surfaces for bonding
  // Skip the last floor index since it's replaced by the roof
  for (let floorIdx = 1; floorIdx < floorHeights.length - 1; floorIdx += 1) {
    // Floor plate center is at floorHeights[floorIdx]
    // Bottom surface at floorHeights[floorIdx] - floorThickness/2
    // Top surface at floorHeights[floorIdx] + floorThickness/2
    const floorY = floorHeights[floorIdx];
    
    const floorFrags = buildFloorFragments(
      width,  // Full width - walls sit on top
      depth,  // Full depth - walls sit on top
      floorThickness,
      fragmentCountPerFloor,
      0, // centerX
      floorY, // centerY
      0, // centerZ
    );
    allFragments.push(...floorFrags);
  }

  // Add a roof plate at the top - this replaces the last floor plate
  // Roof covers the full width including walls
  // Position so bottom of roof touches top of last wall section
  const lastWallTopY = floorHeights[floorCount] - floorThickness * 0.5;
  const roofY = lastWallTopY + floorThickness * 0.5;
  const roofFrags = buildFloorFragments(
    width, // Roof covers the full width including walls
    depth,
    floorThickness,
    fragmentCountPerFloor,
    0, // centerX
    roofY, // centerY at top of tower
    0, // centerZ
  );
  allFragments.push(...roofFrags);

  // Add rectangular foundation under the tower footprint
  addFoundationFragments(
    allFragments,
    width,
    depth,
    foundationHeight,
    groundClearance,
  );

  // Build the scenario from all fragments
  return await buildScenarioFromFragments(
    allFragments,
    { width, depth, height: totalHeight },
    calculatedMass,
    {
      thickness,
      floorThickness,
      columnSize,
      columnsX: numColumnsX,
      columnsZ: numColumnsZ,
      columnSpacing,
      columnInset,
      totalColumns: columnPositions.length,
      floorCount,
      floorHeight,
      fragmentCountPerWall,
      fragmentCountPerFloor,
      fragmentCountPerColumn,
    },
    { useAutoBonding },
  );
}

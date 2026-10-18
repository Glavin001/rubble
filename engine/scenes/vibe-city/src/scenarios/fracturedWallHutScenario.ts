import type { ScenarioDesc } from "@/lib/stress/core/types";
import {
  type FragmentInfo,
  buildWallFragments,
  addFoundationFragments,
  buildScenarioFromFragments,
} from "./fractureUtils";

type FracturedWallHutOptions = {
  /** Width of the hut (X dimension) */
  width?: number;
  /** Depth of the hut (Z dimension) */
  depth?: number;
  /** Height of the walls (Y dimension) */
  height?: number;
  /** Thickness of each wall */
  thickness?: number;
  /** Number of fracture pieces per wall */
  fragmentCountPerWall?: number;
  /** Total mass of the structure */
  deckMass?: number;
};

/**
 * Builds a hut structure from 4 fractured walls (front, back, left, right).
 * Returns fragmentGeometries for auto-bonding support.
 */
export async function buildFracturedWallHutScenario({
  width = 6.5,
  depth = 5.2,
  height = 3.4,
  thickness = 0.32,
  fragmentCountPerWall = 30,
  deckMass = 19_000,
}: FracturedWallHutOptions = {}): Promise<ScenarioDesc> {
  // Foundation dimensions
  const foundationHeight = Math.min(0.08, height * 0.06);
  const groundClearance = Math.max(0.001, foundationHeight * 0.05);
  const foundationClearance = Math.max(0.001, foundationHeight * 0.05);
  const wallLiftY = groundClearance + foundationHeight + foundationClearance;

  // Calculate wall positions (walls form the perimeter of the hut)
  const halfWidth = width * 0.5;
  const halfDepth = depth * 0.5;

  // All fragments from all walls
  const allFragments: FragmentInfo[] = [];

  // Front wall (at z = -halfDepth, spans X)
  const frontWallFrags = buildWallFragments(
    width,
    height,
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
    height,
    thickness,
    fragmentCountPerWall,
    0, // centerX
    halfDepth - thickness * 0.5, // centerZ (inset by half thickness)
    0, // no rotation
    wallLiftY,
  );
  allFragments.push(...backWallFrags);

  // Left wall (at x = -halfWidth, spans Z) - need to account for front/back wall overlap
  const sideWallSpan = depth - thickness * 2; // Subtract thickness on both ends to fit between front/back
  const leftWallFrags = buildWallFragments(
    sideWallSpan,
    height,
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
    height,
    thickness,
    fragmentCountPerWall,
    halfWidth - thickness * 0.5, // centerX (inset by half thickness)
    0, // centerZ
    Math.PI * 0.5, // 90 degree rotation
    wallLiftY,
  );
  allFragments.push(...rightWallFrags);

  // Add rectangular foundation under the entire hut footprint
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
    { width, depth, height },
    deckMass,
    {
      thickness,
      fragmentCountPerWall,
    },
  );
}

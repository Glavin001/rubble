import * as THREE from "three";
import type { ScenarioDesc } from "@/lib/stress/core/types";
import {
  type FragmentInfo,
  buildFloorFragments,
  buildColumnFragments,
  buildScenarioFromFragments,
} from "./fractureUtils";

type FracturedBridgeOptions = {
  /** X length of deck (m) */
  span?: number;
  /** Z width of deck (m) */
  deckWidth?: number;
  /** Y thickness of deck (m) */
  deckThickness?: number;
  /** Distance from ground to deck bottom (m) */
  pierHeight?: number;
  /** Number of vertical posts at each end across Z */
  supportsPerSide?: number;
  /** Cross-section size of each support post */
  postSize?: number;
  /** Thin foundation plate thickness (m) */
  footingThickness?: number;
  /** Number of fracture pieces for the deck */
  fragmentCountPerDeck?: number;
  /** Number of fracture pieces per post */
  fragmentCountPerPost?: number;
  /** Total mass distributed among deck blocks */
  deckMass?: number;
};

/**
 * Builds a beam bridge structure from fractured deck and support posts.
 * Similar to buildBeamBridgeScenario but uses three-pinata fracturing for each component.
 */
export async function buildFracturedBridgeScenario({
  span = 18.0,
  deckWidth = 5.0,
  deckThickness = 0.6,
  pierHeight = 2.8,
  supportsPerSide = 4,
  postSize = 0.4,
  footingThickness = 0.12,
  fragmentCountPerDeck = 40,
  fragmentCountPerPost = 5,
  deckMass = 60_000,
}: FracturedBridgeOptions = {}): Promise<ScenarioDesc> {
  // Foundation and deck positioning
  const groundClearance = 0.001;
  const deckBottomY = groundClearance + footingThickness + pierHeight;
  const deckCenterY = deckBottomY + deckThickness * 0.5;

  // All fragments from deck and supports
  const allFragments: FragmentInfo[] = [];

  // Build the deck (horizontal slab spanning the bridge)
  const deckFrags = buildFloorFragments(
    span,           // spanX
    deckWidth,      // spanZ
    deckThickness,  // thickness
    fragmentCountPerDeck,
    0,              // centerX
    deckCenterY,    // centerY
    0,              // centerZ
  );
  allFragments.push(...deckFrags);

  // Build support posts at each end of the bridge
  // Posts are at X = -span/2 and X = +span/2
  const halfSpan = span * 0.5;
  const halfWidth = deckWidth * 0.5;

  // Calculate Z positions for posts (evenly distributed across deck width)
  const postZPositions: number[] = [];
  for (let i = 0; i < supportsPerSide; i += 1) {
    const t = supportsPerSide === 1 ? 0.5 : i / (supportsPerSide - 1);
    const z = -halfWidth + postSize * 0.5 + t * (deckWidth - postSize);
    postZPositions.push(z);
  }

  // Post positions at each end (left and right)
  const postXPositions = [-halfSpan + postSize * 0.5, halfSpan - postSize * 0.5];

  for (const postX of postXPositions) {
    for (const postZ of postZPositions) {
      // Build the support post (vertical column)
      const postFrags = buildColumnFragments(
        postSize,           // sizeX
        postSize,           // sizeZ
        pierHeight,         // height
        fragmentCountPerPost,
        postX,              // centerX
        groundClearance + footingThickness, // baseY (bottom of post)
        postZ,              // centerZ
      );
      allFragments.push(...postFrags);

      // Add footing under this post (support fragment - mass = 0)
      const footingGeom = new THREE.BoxGeometry(postSize, footingThickness, postSize);
      const footingWorldPos = new THREE.Vector3(
        postX,
        groundClearance + footingThickness * 0.5,
        postZ,
      );
      allFragments.push({
        worldPosition: footingWorldPos,
        halfExtents: new THREE.Vector3(
          postSize * 0.5,
          footingThickness * 0.5,
          postSize * 0.5,
        ),
        geometry: footingGeom,
        isSupport: true,
      });
    }
  }

  // Build the scenario from all fragments
  return await buildScenarioFromFragments(
    allFragments,
    { width: span, depth: deckWidth, height: deckThickness + pierHeight + footingThickness },
    deckMass,
    {
      span,
      deckWidth,
      deckThickness,
      pierHeight,
      supportsPerSide,
      postSize,
      footingThickness,
      fragmentCountPerDeck,
      fragmentCountPerPost,
    },
  );
}


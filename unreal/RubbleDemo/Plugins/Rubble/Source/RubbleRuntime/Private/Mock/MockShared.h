// Shared pieces of the mock backend. This is scripted, ballistic stand-in motion for building and
// tuning the renderers: it is not Rubble's physics and makes no physical claims (no body-body
// contacts; stacks fall when what holds them up wakes; one ground plane at Z = 0).

#pragma once

#include "CoreMinimal.h"
#include "Math/RandomStream.h"
#include "RubbleTypes.h"

namespace RubbleMock
{
	constexpr double GravityCm = 981.0;

	/** One mock rigid body. The solver frame is the centre of mass; the reported transform is the body origin. */
	struct FBody
	{
		FVector Com = FVector::ZeroVector;
		FQuat Rot = FQuat::Identity;
		FVector Vel = FVector::ZeroVector;
		FVector AngVel = FVector::ZeroVector; // rad/s, world
		FVector LocalCom = FVector::ZeroVector; // centre of mass in the origin frame
		FVector HalfExtents = FVector(50.0); // about the centre of mass, for the ground support
		float Radius = 50.0f; // bounding sphere about the centre of mass
		float Mass = 1.0f;
		bool bAwake = false;
		bool bStatic = false;
		bool bKinematic = false; // moved by its scene, never integrated
		bool bProjectile = false;
		bool bWokeThisStep = false;
		bool bAlive = true;
		float RestTime = 0.0f;
		float Age = 0.0f;
		ERubbleMaterial Material = ERubbleMaterial::Concrete;
		uint64 UserData = 0;
		FTransform Reported = FTransform::Identity;
		bool bReportedOnce = false;
		/** Bodies resting on this one: woken when it wakes. */
		TArray<int32> Supports;

		FTransform Origin() const { return FTransform(Rot, Com - Rot.RotateVector(LocalCom)); }
		void SetOrigin(const FTransform& T)
		{
			Rot = T.GetRotation();
			Com = T.TransformPosition(LocalCom);
		}
	};

	/** Ballistic bodies on a ground plane, with sleep, support chains and projectile blasts. */
	class FWorld
	{
	public:
		TArray<FBody> Bodies;
		FRandomStream Random{ 1234 };
		float Restitution = 0.25f;
		float BlastRadius = 220.0f;
		float BlastSpeed = 900.0f;
		/** Emit an Impact when a body lands faster than this (cm/s). */
		float ImpactSpeed = 350.0f;
		/** Called when a projectile hits a body (index) at a point; the scene may answer (fracture). */
		TFunction<bool(int32 Projectile, int32 Hit, const FVector& Point)> OnProjectileHit;

		static FRubbleBodyId IdOf(int32 Index) { return FRubbleBodyId::Make(uint32(Index), 1); }

		int32 Add(const FBody& Body);
		void Wake(int32 Index, const FVector& DeltaVel = FVector::ZeroVector);
		/** Wake everything within Radius of Point, pushed away from it, falling off with distance. */
		void Blast(const FVector& Point, float Radius, float Speed, const FVector& Along);
		/** Bodies resting on others get support links (built once, for stacks). */
		void BuildSupports();
		int32 Launch(const FVector& Origin, const FVector& Direction, float Speed, float Radius);
		void Step(float Dt, FRubbleFrame& Out);
		/** Report Index at its current pose in this frame (a host-made move: teleport, kinematic). */
		void Report(int32 Index, FRubbleFrame& Out, bool bFellAsleep = false);

		/** The ground under P within Radius: Z = 0 plus whatever has come to rest there. */
		double GroundAt(const FVector& P, double Radius) const;

	private:
		/** Rest heights on a 40 cm grid: a body that sleeps raises the cells under it, so piles stack. */
		TMap<FIntPoint, float> Heights;
		struct FRaised
		{
			TArray<TPair<FIntPoint, float>> Cells; // cell, height before
			float Top = 0.0f;
		};
		TMap<int32, FRaised> RaisedBy;
		void Settle(int32 Index);
		void Unsettle(int32 Index);
		TArray<int32> Awake;
		TArray<int32> Woken;
		TArray<int32> Projectiles;
	};

	// ----- geometry --------------------------------------------------------------------------

	/** A box mesh of half extents H centred at C (normals, box UVs; Unreal winding). */
	void AppendBox(FRubbleMeshData& Mesh, const FVector3f& H, const FVector3f& C = FVector3f::ZeroVector, ERubbleMaterial Material = ERubbleMaterial::Concrete);

	/** A convex polytope: vertices and planar faces (counter-clockwise seen from outside). */
	struct FPolytope
	{
		TArray<FVector3d> Vertices;
		TArray<TArray<int32>> Faces;
		TArray<bool> FaceInterior; // a face made by a cut
		static FPolytope Box(const FVector3d& Min, const FVector3d& Max);
		/** Keep the half-space N . x <= D; the new face is interior. */
		void Clip(const FVector3d& N, double D);
		FVector3d Centroid() const;
		bool IsEmpty() const { return Vertices.Num() < 4 || Faces.Num() < 4; }
	};

	/** Triangulated mesh of a polytope, relative to Origin. Exterior faces get Exterior, cuts Interior. */
	void PolytopeMesh(const FPolytope& P, const FVector3d& Origin, ERubbleMaterial Exterior, FRubbleMeshData& Out);

	/** Unit-ish convex hull: a random-ish rock (an icosahedron, jittered), for hull bodies. */
	void AppendRock(FRubbleMeshData& Mesh, float Radius, FRandomStream& Random);

	/** Smooth vertex normals of an indexed mesh (area weighted). */
	void ComputeNormals(const TArray<FVector3f>& Positions, const TArray<uint32>& Indices, TArray<FVector3f>& Normals);
}

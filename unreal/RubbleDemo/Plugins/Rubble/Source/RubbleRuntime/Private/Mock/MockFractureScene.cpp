// The fracture lab's scenes as mock destructibles. Each structure is a set of Voronoi chunks (box
// cells clipped by their neighbours' bisector planes; cut faces are Interior) drawn as one root
// body until it is hit. A hit breaks the chunks near it; the rest splits into connected components:
// what still reaches an anchor keeps the body, every other component becomes a new body that
// falls. Chunk world = body x rest, with every body's origin at the destructible's frame (the
// render contract asked of the API). The ids are engine-fracture samples.rs's.

#include "Mock/MockScene.h"

namespace RubbleMock
{
	namespace
	{
		FRubbleDemoDesc Fracture(const TCHAR* Id, const TCHAR* Title);

		FRubbleDemoDesc Town()
		{
			FRubbleDemoDesc D = Fracture(TEXT("town"), TEXT("Vibe Town: 70 buildings, 52k chunks (vibe-land)"));
			D.Params.Add({ TEXT("strength"), TEXT("Bond strength (x the pack's)"), 1.0f, 0.25f, 4.0f });
			return D;
		}

		FRubbleDemoDesc GlassTower()
		{
			FRubbleDemoDesc D = Fracture(TEXT("glasstower"), TEXT("Glass tower and pavilion (engine-fracture glass_tower.rs)"));
			D.Params.Add({ TEXT("strength"), TEXT("Bond strength (x the pack's)"), 1.0f, 0.25f, 4.0f });
			D.Params.Add({ TEXT("threshold"), TEXT("Threshold model with support loss (0: bond joints, the fracture lab's)"), 0.0f, 0.0f, 1.0f });
			return D;
		}

		FRubbleDemoDesc Fracture(const TCHAR* Id, const TCHAR* Title)
		{
			FRubbleDemoDesc D;
			D.Family = ERubbleFamily::Fracture;
			D.Id = Id;
			D.Title = Title;
			D.Params = { { TEXT("autoshot"), TEXT("Auto shot after (s, 0 off)"), 1.5f, 0, 10 } };
			D.StepHz = 60.0;
			return D;
		}
	}

	TArray<FRubbleDemoDesc> FractureCatalog()
	{
		return {
			Fracture(TEXT("house"), TEXT("Veneer house")),
			Fracture(TEXT("cuboid"), TEXT("Cuboid")),
			Fracture(TEXT("arch"), TEXT("Masonry arch")),
			Fracture(TEXT("row"), TEXT("Row of cuboids")),
			Fracture(TEXT("grid"), TEXT("Grid of cuboids")),
			Fracture(TEXT("cuboidGrid"), TEXT("Cuboid grid")),
			Fracture(TEXT("suburb"), TEXT("Suburb: a street of houses")),
			Fracture(TEXT("cuboids"), TEXT("Cuboids")),
			Fracture(TEXT("cuboids2x"), TEXT("Cuboids ×2")),
			Town(),
			GlassTower(),
		};
	}

	namespace
	{
		struct FChunk
		{
			int32 Destructible = 0;
			int32 Piece = 0;                          // which slab of the destructible
			FIntVector Cell = FIntVector::ZeroValue;  // grid cell within its piece, for adjacency
			FVector Center = FVector::ZeroVector;     // in the destructible's frame
			float Size = 30.0f;
			float Mass = 10.0f;
			bool bAnchor = false;
			/** Chunks of other pieces it rests on or holds (the roof on the walls). */
			TArray<int32> Links;
		};

		struct FActor
		{
			int32 Destructible = 0;
			TArray<int32> Chunks;
			bool bAnchored = false;
		};

		class FFractureScene : public FScene
		{
		public:
			bool Build(const FRubbleSceneRequest& Request, FRubbleSceneAssets& Assets, FString& Error) override;
			void Step(float Dt, FRubbleFrame& Out) override;
			void Launch(const FVector& Origin, const FVector& Direction) override;

		private:
			FWorld World;
			FRubbleSceneAssets* A = nullptr;
			TArray<FChunk> Chunks;
			TMap<int32, FActor> Actors; // by body index
			TArray<int32> Balls;
			int32 NextBall = 0;
			float Time = 0.0f;
			float AutoShotAt = -1.0f;
			TArray<FRubbleEvent> Pending;

			int32 BeginDestructible(const FTransform& Placement);
			void Slab(int32 D, int32 Piece, const FVector& Min, const FVector& Max, const FIntVector& N, ERubbleMaterial Material, bool bAnchoredBottom,
				TFunctionRef<bool(const FVector&)> Keep);
			void ArchBlocks(int32 D);
			void FinishDestructible(int32 D);
			void House(const FVector& At, float Yaw);
			void Cuboid(const FVector& At, float Half, int32 Cells, float Yaw);
			bool Adjacent(int32 a, int32 b) const;
			FRubbleActorInfo Info(int32 Body) const;
			bool Hit(int32 Body, const FVector& Point, const FVector& Velocity);
			int32 NewActor(int32 D, TArray<int32>&& Members, bool bAnchored, const FVector& Vel, const FVector& AngVel);
		};

		int32 FFractureScene::BeginDestructible(const FTransform& Placement)
		{
			FRubbleDestructibleVisual V;
			V.Placement = Placement;
			return A->Destructibles.Add(V);
		}

		void FFractureScene::Slab(int32 D, int32 Piece, const FVector& Min, const FVector& Max, const FIntVector& N, ERubbleMaterial Material, bool bAnchoredBottom,
			TFunctionRef<bool(const FVector&)> Keep)
		{
			const FVector Size = Max - Min;
			const FVector Cell(Size.X / N.X, Size.Y / N.Y, Size.Z / N.Z);
			// Jittered seeds, one per cell.
			TArray<FVector> Seeds;
			Seeds.SetNum(N.X * N.Y * N.Z);
			auto Index = [&N](int32 i, int32 j, int32 k) { return (k * N.Y + j) * N.X + i; };
			for (int32 k = 0; k < N.Z; ++k)
			{
				for (int32 j = 0; j < N.Y; ++j)
				{
					for (int32 i = 0; i < N.X; ++i)
					{
						const FVector Jitter(World.Random.FRandRange(0.2f, 0.8f), World.Random.FRandRange(0.2f, 0.8f), World.Random.FRandRange(0.2f, 0.8f));
						Seeds[Index(i, j, k)] = Min + (FVector(i, j, k) + Jitter) * Cell;
					}
				}
			}
			for (int32 k = 0; k < N.Z; ++k)
			{
				for (int32 j = 0; j < N.Y; ++j)
				{
					for (int32 i = 0; i < N.X; ++i)
					{
						const FVector S = Seeds[Index(i, j, k)];
						if (!Keep(S))
						{
							continue;
						}
						FPolytope P = FPolytope::Box(FVector3d(Min), FVector3d(Max));
						for (int32 dk = -2; dk <= 2; ++dk)
						{
							for (int32 dj = -2; dj <= 2; ++dj)
							{
								for (int32 di = -2; di <= 2; ++di)
								{
									const int32 ii = i + di, jj = j + dj, kk = k + dk;
									if ((di | dj | dk) == 0 || ii < 0 || jj < 0 || kk < 0 || ii >= N.X || jj >= N.Y || kk >= N.Z)
									{
										continue;
									}
									const FVector O = Seeds[Index(ii, jj, kk)];
									const FVector3d Nrm = FVector3d(O - S).GetSafeNormal();
									P.Clip(Nrm, FVector3d::DotProduct(Nrm, FVector3d(0.5 * (O + S))));
								}
							}
						}
						if (P.IsEmpty())
						{
							continue;
						}
						const FVector3d C = P.Centroid();
						FRubbleMeshData Mesh;
						PolytopeMesh(P, C, Material, Mesh);
						FRubbleChunkVisual V;
						V.Mesh = A->Meshes.Add(MoveTemp(Mesh));
						V.Destructible = D;
						V.Rest = FTransform(FVector(C));
						const int32 c = A->Chunks.Add(V);
						FChunk Chunk;
						Chunk.Destructible = D;
						Chunk.Piece = Piece;
						Chunk.Cell = FIntVector(i, j, k);
						Chunk.Center = FVector(C);
						Chunk.Size = float(Cell.GetMax());
						Chunk.Mass = float(Cell.X * Cell.Y * Cell.Z * 1e-6 * 2000.0);
						Chunk.bAnchor = bAnchoredBottom && k == 0;
						check(Chunks.Num() == c);
						Chunks.Add(Chunk);
						A->Destructibles[D].Chunks.Add(c);
					}
				}
			}
		}

		void FFractureScene::ArchBlocks(int32 D)
		{
			// Two pillars and a semicircular arch of voussoirs, each a chunk (masonry, no Voronoi).
			const float Span = 400.0f, Rise = 200.0f, Depth = 60.0f;
			auto Block = [&](const FVector& At, const FQuat& Q, const FVector& Half, int32 Seq, bool bAnchor) {
				FRubbleMeshData Mesh;
				Mesh.Material = ERubbleMaterial::Concrete;
				AppendBox(Mesh, FVector3f(Half));
				FRubbleChunkVisual V;
				V.Mesh = A->Meshes.Add(MoveTemp(Mesh));
				V.Destructible = D;
				V.Rest = FTransform(Q, At);
				const int32 c = A->Chunks.Add(V);
				FChunk Chunk;
				Chunk.Destructible = D;
				Chunk.Cell = FIntVector(Seq, 0, 0);
				Chunk.Center = At;
				Chunk.Size = float(Half.GetMax() * 2.0);
				Chunk.Mass = float(8.0 * Half.X * Half.Y * Half.Z * 1e-6 * 2400.0);
				Chunk.bAnchor = bAnchor;
				Chunks.Add(Chunk);
				A->Destructibles[D].Chunks.Add(c);
			};
			int32 Seq = 0;
			for (int32 Side = -1; Side <= 1; Side += 2)
			{
				for (int32 h = 0; h < 6; ++h)
				{
					Block(FVector(0, Side * (Span * 0.5f + 40.0f), 30.0f + h * 60.0f), FQuat::Identity, FVector(Depth * 0.5f, 40, 29), Side < 0 ? Seq++ : 1000 + h, h == 0);
				}
			}
			const int32 Voussoirs = 15;
			for (int32 v = 0; v < Voussoirs; ++v)
			{
				const float a = UE_PI * (v + 0.5f) / Voussoirs;
				const FVector At(0, -FMath::Cos(a) * (Span * 0.5f + 40.0f), 360.0f + FMath::Sin(a) * Rise);
				Block(At, FQuat(FVector(1, 0, 0), a - UE_HALF_PI), FVector(Depth * 0.5f, 26, 38), 6 + v, false);
			}
			// The right pillar's sequence numbers continue after the arch.
			for (FChunk& C : Chunks)
			{
				if (C.Destructible == D && C.Cell.X >= 1000)
				{
					C.Cell.X = 6 + Voussoirs + (5 - (C.Cell.X - 1000));
				}
			}
		}

		bool FFractureScene::Adjacent(int32 a, int32 b) const
		{
			const FChunk& A1 = Chunks[a];
			const FChunk& B1 = Chunks[b];
			if (A1.Destructible != B1.Destructible)
			{
				return false;
			}
			if (A1.Piece != B1.Piece)
			{
				return A1.Links.Contains(b);
			}
			const FIntVector D = A1.Cell - B1.Cell;
			return FMath::Abs(D.X) <= 1 && FMath::Abs(D.Y) <= 1 && FMath::Abs(D.Z) <= 1;
		}

		void FFractureScene::FinishDestructible(int32 D)
		{
			FRubbleDestructibleVisual& V = A->Destructibles[D];
			// The root body: kinematic (anchored), hittable through its bounding sphere.
			FBody B;
			FVector Com = FVector::ZeroVector;
			float Mass = 0.0f;
			FBox Box(ForceInit);
			for (const int32 c : V.Chunks)
			{
				Com += Chunks[c].Center * Chunks[c].Mass;
				Mass += Chunks[c].Mass;
				Box += Chunks[c].Center;
			}
			Com /= FMath::Max(Mass, 1e-3f);
			B.LocalCom = Com;
			B.SetOrigin(V.Placement);
			B.Radius = float(Box.GetExtent().Size() + 60.0);
			B.HalfExtents = Box.GetExtent();
			B.Mass = Mass;
			B.bKinematic = true;
			const int32 Index = World.Add(B);
			V.RootBody = FWorld::IdOf(Index);
			FRubbleBodyVisual BV;
			BV.Id = V.RootBody;
			BV.Initial = V.Placement;
			BV.FirstShape = A->Shapes.Num();
			BV.NumShapes = 0;
			check(A->Bodies.Num() == Index);
			A->Bodies.Add(BV);
			FActor Actor;
			Actor.Destructible = D;
			Actor.Chunks = V.Chunks;
			Actor.bAnchored = true;
			Actors.Add(Index, MoveTemp(Actor));
		}

		void FFractureScene::House(const FVector& At, float Yaw)
		{
			const FTransform Placement(FQuat(FVector::UpVector, Yaw), At);
			const int32 D = BeginDestructible(Placement);
			const float L = 800.0f, W = 600.0f, H = 300.0f, T = 25.0f;
			auto Door = [](const FVector& P) { return !(FMath::Abs(P.Y) < 60.0 && P.Z < 210.0); };
			auto Window = [](const FVector& P) { return !(FMath::Abs(FMath::Abs(P.X) - 220.0) < 70.0 && P.Z > 100.0 && P.Z < 220.0); };
			const int32 First = Chunks.Num();
			// Front and back (along X at Y = +-W/2) with windows, gable ends with the door.
			Slab(D, 0, FVector(-L * 0.5, -W * 0.5, 0), FVector(L * 0.5, -W * 0.5 + T, H), FIntVector(12, 1, 5), ERubbleMaterial::Brick, true, Window);
			Slab(D, 1, FVector(-L * 0.5, W * 0.5 - T, 0), FVector(L * 0.5, W * 0.5, H), FIntVector(12, 1, 5), ERubbleMaterial::Brick, true, Window);
			Slab(D, 2, FVector(-L * 0.5, -W * 0.5 + T, 0), FVector(-L * 0.5 + T, W * 0.5 - T, H), FIntVector(1, 8, 5), ERubbleMaterial::Brick, true, Door);
			Slab(D, 3, FVector(L * 0.5 - T, -W * 0.5 + T, 0), FVector(L * 0.5, W * 0.5 - T, H), FIntVector(1, 8, 5), ERubbleMaterial::Brick, true, [](const FVector&) { return true; });
			const int32 Walls = Chunks.Num();
			// A flat roof slab, resting on the walls' top row.
			Slab(D, 4, FVector(-L * 0.5 - 30, -W * 0.5 - 30, H), FVector(L * 0.5 + 30, W * 0.5 + 30, H + 22), FIntVector(10, 8, 1), ERubbleMaterial::Concrete, false, [](const FVector&) { return true; });
			for (int32 r = Walls; r < Chunks.Num(); ++r)
			{
				for (int32 w = First; w < Walls; ++w)
				{
					if (Chunks[w].Cell.Z == 4 && FVector::Dist2D(Chunks[w].Center, Chunks[r].Center) < 140.0)
					{
						Chunks[r].Links.Add(w);
						Chunks[w].Links.Add(r);
					}
				}
			}
			FinishDestructible(D);
		}

		void FFractureScene::Cuboid(const FVector& At, float Half, int32 Cells, float Yaw)
		{
			const int32 D = BeginDestructible(FTransform(FQuat(FVector::UpVector, Yaw), At));
			Slab(D, 0, FVector(-Half, -Half, 0), FVector(Half, Half, 2.0 * Half), FIntVector(Cells), ERubbleMaterial::Concrete, true, [](const FVector&) { return true; });
			FinishDestructible(D);
		}

		bool FFractureScene::Build(const FRubbleSceneRequest& R, FRubbleSceneAssets& Assets, FString& Error)
		{
			A = &Assets;
			Assets.Stage = ERubbleStage::Lot;
			Assets.GroundHalfExtentCm = 8000.0f;
			const FString& Id = R.Id;
			FVector Focus(0, 0, 150);
			float Distance = 1600.0f;
			if (Id == TEXT("town"))
			{
				// The ABI builds it from the pack; the layout is its stage and camera only.
				Assets.Stage = ERubbleStage::Town;
				// Out to the fog: the plane's edge is never seen.
				Assets.GroundHalfExtentCm = 400000.0f;
				Focus = FVector(0, 0, 600);
				Distance = 16000.0f;
			}
			else if (Id == TEXT("glasstower"))
			{
				// Built by the ABI from its pack, on the town's stage.
				Assets.Stage = ERubbleStage::Town;
				Assets.GroundHalfExtentCm = 400000.0f;
				Focus = FVector(0, -1000, 1200);
				Distance = 7000.0f;
			}
			else if (Id == TEXT("house"))
			{
				House(FVector::ZeroVector, 0.0f);
			}
			else if (Id == TEXT("suburb"))
			{
				for (int32 i = 0; i < 3; ++i)
				{
					for (int32 j = 0; j < 3; ++j)
					{
						House(FVector((i - 1) * 1500.0, (j - 1) * 1300.0, 0), (i + j) % 2 ? 0.0f : UE_HALF_PI * 0.0f);
					}
				}
				Distance = 4200.0f;
			}
			else if (Id == TEXT("cuboid"))
			{
				Cuboid(FVector::ZeroVector, 100.0f, 5, 0.0f);
				Distance = 900.0f;
				Focus.Z = 100.0;
			}
			else if (Id == TEXT("arch"))
			{
				const int32 D = BeginDestructible(FTransform::Identity);
				ArchBlocks(D);
				FinishDestructible(D);
				Focus.Z = 250.0;
			}
			else if (Id == TEXT("row") || Id == TEXT("cuboids") || Id == TEXT("cuboids2x"))
			{
				const int32 N = Id == TEXT("cuboids2x") ? 10 : (Id == TEXT("cuboids") ? 5 : 6);
				for (int32 i = 0; i < N; ++i)
				{
					Cuboid(FVector(0, (i - N * 0.5 + 0.5) * 320.0, 0), 80.0f, 4, World.Random.FRandRange(-0.3f, 0.3f));
				}
				Distance = 300.0f * N + 600.0f;
			}
			else if (Id == TEXT("grid") || Id == TEXT("cuboidGrid"))
			{
				const int32 N = Id == TEXT("grid") ? 4 : 3;
				for (int32 i = 0; i < N; ++i)
				{
					for (int32 j = 0; j < N; ++j)
					{
						Cuboid(FVector((i - N * 0.5 + 0.5) * 400.0, (j - N * 0.5 + 0.5) * 400.0, 0), Id == TEXT("grid") ? 70.0f : 120.0f, 4, 0.0f);
					}
				}
				Distance = 2600.0f;
			}
			else
			{
				Error = FString::Printf(TEXT("no fracture scene '%s'"), *Id);
				return false;
			}
			FBodyBuilder B{ World, Assets };
			for (int32 i = 0; i < 12; ++i)
			{
				Balls.Add(B.ParkedBall(25.0f));
			}
			const float Auto = R.Param(TEXT("autoshot"), 1.5f);
			AutoShotAt = Auto > 0.0f ? Auto : -1.0f;
			Assets.CameraTarget = Focus;
			Assets.CameraEye = Focus + FVector(-Distance, -0.55 * Distance, 0.45 * Distance);
			World.OnProjectileHit = [this](int32 Projectile, int32 Hit, const FVector& Point) {
				return this->Hit(Hit, World.Bodies[Projectile].Com, World.Bodies[Projectile].Vel);
			};
			return true;
		}

		FRubbleActorInfo FFractureScene::Info(int32 Body) const
		{
			FRubbleActorInfo I;
			I.Body = FWorld::IdOf(Body);
			const FActor& Actor = Actors.FindChecked(Body);
			I.Chunks = Actor.Chunks;
			I.bAnchored = Actor.bAnchored;
			const FBody& B = World.Bodies[Body];
			I.Mass = B.Mass;
			I.Com = B.Com;
			I.LinearVelocity = B.Vel;
			return I;
		}

		int32 FFractureScene::NewActor(int32 D, TArray<int32>&& Members, bool bAnchored, const FVector& Vel, const FVector& AngVel)
		{
			const FTransform& Placement = A->Destructibles[D].Placement;
			FBody B;
			FVector Com = FVector::ZeroVector;
			float Mass = 0.0f;
			FBox Box(ForceInit);
			for (const int32 c : Members)
			{
				Com += Chunks[c].Center * Chunks[c].Mass;
				Mass += Chunks[c].Mass;
				Box += Chunks[c].Center;
			}
			Com /= FMath::Max(Mass, 1e-3f);
			const float Pad = Chunks[Members[0]].Size * 0.5f;
			B.LocalCom = Com;
			B.SetOrigin(Placement);
			B.HalfExtents = Box.GetExtent() + FVector(Pad);
			B.Radius = float(B.HalfExtents.Size());
			B.Mass = Mass;
			B.Material = A->Meshes[A->Chunks[Members[0]].Mesh].Material;
			B.bKinematic = bAnchored;
			const int32 Index = World.Add(B);
			if (!bAnchored)
			{
				World.Wake(Index, Vel);
				World.Bodies[Index].AngVel = AngVel;
			}
			FActor Actor;
			Actor.Destructible = D;
			Actor.Chunks = MoveTemp(Members);
			Actor.bAnchored = bAnchored;
			Actors.Add(Index, MoveTemp(Actor));
			return Index;
		}

		bool FFractureScene::Hit(int32 Body, const FVector& Point, const FVector& Velocity)
		{
			FActor* Actor = Actors.Find(Body);
			if (!Actor)
			{
				return false;
			}
			// The hit point in the destructible's frame (every body's origin is that frame).
			const FTransform BodyT = World.Bodies[Body].Origin();
			const FVector Local = BodyT.InverseTransformPosition(Point);
			const double Reach = 30.0 + 0.04 * Velocity.Size();
			TSet<int32> Broken;
			double Nearest = 1e12;
			for (const int32 c : Actor->Chunks)
			{
				const double Dist = FVector::Dist(Chunks[c].Center, Local) - Chunks[c].Size * 0.5;
				Nearest = FMath::Min(Nearest, Dist);
				if (Dist < Reach)
				{
					Broken.Add(c);
				}
			}
			if (Nearest > 25.0 + 0.6 * Chunks[Actor->Chunks[0]].Size)
			{
				return false; // inside the bounding sphere, not at a chunk yet
			}
			if (Broken.Num() == 0)
			{
				return true;
			}
			const int32 D = Actor->Destructible;
			const FRubbleActorInfo Parent = Info(Body);
			FRubbleEvent Split;
			Split.Type = ERubbleEventType::Split;
			Split.Destructible = D;
			Split.Position = Point;
			Split.Parent = Parent;

			// Bonds severed: every adjacency between a broken chunk and any other chunk.
			int32 Bonds = 0;
			TArray<int32> Remaining;
			for (const int32 c : Actor->Chunks)
			{
				if (!Broken.Contains(c))
				{
					Remaining.Add(c);
				}
			}
			for (const int32 b : Broken)
			{
				for (const int32 c : Actor->Chunks)
				{
					Bonds += (c != b && Adjacent(b, c)) ? 1 : 0;
				}
			}

			// Components of what is left: flood fill over adjacency.
			TArray<TArray<int32>> Components;
			TSet<int32> Seen;
			for (const int32 Start : Remaining)
			{
				if (Seen.Contains(Start))
				{
					continue;
				}
				TArray<int32> Comp, Stack{ Start };
				Seen.Add(Start);
				while (Stack.Num())
				{
					const int32 c = Stack.Pop(EAllowShrinking::No);
					Comp.Add(c);
					for (const int32 o : Remaining)
					{
						if (!Seen.Contains(o) && Adjacent(c, o))
						{
							Seen.Add(o);
							Stack.Add(o);
						}
					}
				}
				Components.Add(MoveTemp(Comp));
			}
			// The body stays with the anchored component (or the largest, for a loose piece).
			int32 Keep = INDEX_NONE;
			for (int32 i = 0; i < Components.Num(); ++i)
			{
				const bool bAnchored = Actor->bAnchored && Components[i].ContainsByPredicate([this](int32 c) { return Chunks[c].bAnchor; });
				if (bAnchored || (!Actor->bAnchored && (Keep == INDEX_NONE || Components[i].Num() > Components[Keep].Num())))
				{
					if (Keep == INDEX_NONE || bAnchored)
					{
						Keep = i;
					}
				}
			}
			const FVector Dir = Velocity.GetSafeNormal();
			const bool bWasAnchored = Actor->bAnchored;
			if (Keep != INDEX_NONE)
			{
				Actor->Chunks = Components[Keep];
			}
			else
			{
				Actor->Chunks.Reset();
			}
			// Actor may dangle after NewActor adds to the map: no use past here.
			Actor = nullptr;
			FBody& Kept = World.Bodies[Body];
			if (Keep != INDEX_NONE)
			{
				// Re-centre the kept body's mass on its remaining chunks; its origin stays put.
				const FTransform Origin = Kept.Origin();
				FVector Com = FVector::ZeroVector;
				float Mass = 0.0f;
				for (const int32 c : Components[Keep])
				{
					Com += Chunks[c].Center * Chunks[c].Mass;
					Mass += Chunks[c].Mass;
				}
				Kept.LocalCom = Com / FMath::Max(Mass, 1e-3f);
				Kept.Mass = Mass;
				Kept.SetOrigin(Origin);
				if (!bWasAnchored)
				{
					World.Wake(Body, Dir * 200.0f);
				}
				Split.Children.Add(Info(Body));
			}
			else
			{
				World.Bodies[Body].bAlive = false;
			}
			for (int32 i = 0; i < Components.Num(); ++i)
			{
				if (i != Keep)
				{
					const int32 New = NewActor(D, MoveTemp(Components[i]), false, Dir * 120.0f + World.Random.VRand() * 60.0f, World.Random.VRand() * 0.8f);
					Split.Children.Add(Info(New));
				}
			}
			for (const int32 b : Broken)
			{
				const FVector Out = (Chunks[b].Center - Local).GetSafeNormal();
				TArray<int32> One{ b };
				const int32 New = NewActor(D, MoveTemp(One), false,
					BodyT.TransformVector(Out) * World.Random.FRandRange(150.0f, 450.0f) + Dir * Velocity.Size() * World.Random.FRandRange(0.15f, 0.35f) + FVector(0, 0, 150),
					World.Random.VRand() * World.Random.FRandRange(2.0f, 9.0f));
				Split.Children.Add(Info(New));
			}
			FRubbleEvent Bond;
			Bond.Type = ERubbleEventType::BondsBroken;
			Bond.Destructible = D;
			Bond.Count = Bonds;
			Bond.Position = Point;
			Pending.Add(Bond);
			Pending.Add(MoveTemp(Split));
			FRubbleEvent Impact;
			Impact.Type = ERubbleEventType::Impact;
			Impact.Position = Point;
			Impact.Normal = -Dir;
			Impact.Magnitude = float(Velocity.Size());
			Impact.MaterialA = ERubbleMaterial::Steel;
			Impact.MaterialB = ERubbleMaterial::Brick;
			Pending.Add(Impact);
			return true;
		}

		void FFractureScene::Launch(const FVector& Origin, const FVector& Direction)
		{
			if (Balls.Num())
			{
				FBodyBuilder::Fire(World, Balls[NextBall], Origin, Direction, 3500.0f);
				NextBall = (NextBall + 1) % Balls.Num();
			}
		}

		void FFractureScene::Step(float Dt, FRubbleFrame& Out)
		{
			Time += Dt;
			if (AutoShotAt >= 0.0f && Time >= AutoShotAt)
			{
				AutoShotAt = -1.0f;
				const FVector Aim = A->CameraTarget + FVector(0, 0, 30);
				const FVector From = A->CameraEye;
				const double T = (Aim - From).Size() / 3500.0;
				Launch(From, Aim - From + FVector(0, 0, 0.5 * GravityCm * T * T));
			}
			World.Step(Dt, Out);
			Out.Events.Append(MoveTemp(Pending));
			Pending.Reset();
		}
	}

	TUniquePtr<FScene> MakeFractureScene()
	{
		return TUniquePtr<FScene>(new FFractureScene());
	}
}

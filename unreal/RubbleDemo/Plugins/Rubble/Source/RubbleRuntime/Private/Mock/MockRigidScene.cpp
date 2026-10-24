// demo.html's rigid scenes as mock layouts: stacks that rest until something wakes them, piles and
// grids that drop, kinematic chains and motors that always move. The ids and titles are the page's.

#include "Mock/MockScene.h"

namespace RubbleMock
{
	namespace
	{
		FRubbleDemoDesc Rigid(const TCHAR* Id, const TCHAR* Title, TArray<FRubbleParamDesc> Params = {})
		{
			FRubbleDemoDesc D;
			D.Family = ERubbleFamily::Rigid;
			D.Id = Id;
			D.Title = Title;
			D.Params = MoveTemp(Params);
			D.StepHz = 60.0;
			return D;
		}
	}

	TArray<FRubbleDemoDesc> RigidCatalog()
	{
		return {
			Rigid(TEXT("wall"), TEXT("wall smash 2k, 10 iterations")),
			Rigid(TEXT("ring"), TEXT("brick ring 9k, 4 iterations")),
			Rigid(TEXT("gables"), TEXT("brick gables 6.7k, 3 iterations")),
			Rigid(TEXT("pyramid"), TEXT("settled pyramid"), { { TEXT("levels"), TEXT("Levels"), 30, 2, 50 } }),
			Rigid(TEXT("pile"), TEXT("settled pile 32k, 4 iterations")),
			Rigid(TEXT("columns"), TEXT("box columns 100k, 4 iterations")),
			Rigid(TEXT("ring110k"), TEXT("brick ring 110k, 4 iterations")),
			Rigid(TEXT("gables506k"), TEXT("brick gables 506k, 3 iterations")),
			Rigid(TEXT("joints"), TEXT("joints showcase")),
			Rigid(TEXT("ragdolls"), TEXT("ragdolls 40")),
			Rigid(TEXT("ragdolls10k"), TEXT("ragdoll pile 11k bodies, 10k joints")),
			Rigid(TEXT("hingechains10k"), TEXT("hinge chains 10k bodies, 10k joints")),
			Rigid(TEXT("motorfield10k"), TEXT("motor field 10k bodies, 2.5k motors")),
			Rigid(TEXT("legacy-empty"), TEXT("legacy · empty")),
			Rigid(TEXT("legacy-ground"), TEXT("legacy · ground")),
			Rigid(TEXT("legacy-dynamic-friction"), TEXT("legacy · dynamic friction (μ 5 → 0)")),
			Rigid(TEXT("legacy-static-friction"), TEXT("legacy · static friction ramp (μ 0.25 → 0.5)")),
			Rigid(TEXT("legacy-pyramid"), TEXT("legacy · pyramid 136")),
			Rigid(TEXT("legacy-stack"), TEXT("legacy · stack 10")),
			Rigid(TEXT("legacy-stack-ratio"), TEXT("legacy · stack ratio 1:8")),
			Rigid(TEXT("legacy-scatter"), TEXT("legacy · scatter 3k (spheres, boxes, capsules)")),
			Rigid(TEXT("legacy-scatter-boxes"), TEXT("legacy · scatter boxes 4k")),
			Rigid(TEXT("legacy-grid-boxes"), TEXT("legacy · grid boxes 1.5k")),
			Rigid(TEXT("legacy-slanted-grid"), TEXT("legacy · slanted grid 3.2k")),
			Rigid(TEXT("legacy-grid10k"), TEXT("legacy · grid 10k")),
			Rigid(TEXT("legacy-grid100k"), TEXT("grid 100k (legacy grid ×10)")),
		};
	}

	// ----- body builder ------------------------------------------------------------------------

	int32 FBodyBuilder::Add(FBody& Body, const FRubbleShapeVisual& Shape, bool bStatic)
	{
		Body.bStatic = bStatic;
		const int32 Index = World.Add(Body);
		FRubbleBodyVisual V;
		V.Id = FWorld::IdOf(Index);
		V.Initial = Body.Origin();
		V.FirstShape = Assets.Shapes.Add(Shape);
		V.NumShapes = 1;
		V.bStatic = bStatic;
		V.bAwake = Body.bAwake;
		check(Assets.Bodies.Num() == Index);
		Assets.Bodies.Add(V);
		World.Bodies[Index].UserData = uint64(Index);
		return Index;
	}

	int32 FBodyBuilder::Box(const FVector& Center, const FVector& Half, const FQuat& Rot, ERubbleMaterial Material, const FLinearColor& Tint, bool bStatic, bool bAwake)
	{
		FBody B;
		B.Com = Center;
		B.Rot = Rot;
		B.HalfExtents = Half;
		B.Radius = float(Half.Size());
		B.Mass = float(8.0 * Half.X * Half.Y * Half.Z * 1e-6 * 2000.0);
		B.bAwake = bAwake && !bStatic;
		B.Material = Material;
		FRubbleShapeVisual S;
		S.Kind = ERubbleShape::Box;
		S.HalfExtents = FVector3f(Half);
		S.Material = Material;
		S.Tint = Tint;
		return Add(B, S, bStatic);
	}

	int32 FBodyBuilder::Sphere(const FVector& Center, float Radius, ERubbleMaterial Material, const FLinearColor& Tint, bool bAwake)
	{
		FBody B;
		B.Com = Center;
		B.HalfExtents = FVector(Radius);
		B.Radius = Radius;
		B.bAwake = bAwake;
		B.Material = Material;
		FRubbleShapeVisual S;
		S.Kind = ERubbleShape::Sphere;
		S.Radius = Radius;
		S.Material = Material;
		S.Tint = Tint;
		return Add(B, S, false);
	}

	int32 FBodyBuilder::Capsule(const FVector& Center, float Radius, float HalfHeight, const FQuat& Rot, ERubbleMaterial Material, const FLinearColor& Tint, bool bAwake)
	{
		FBody B;
		B.Com = Center;
		B.Rot = Rot;
		B.HalfExtents = FVector(Radius, Radius, Radius + HalfHeight);
		B.Radius = Radius + HalfHeight;
		B.bAwake = bAwake;
		B.Material = Material;
		FRubbleShapeVisual S;
		S.Kind = ERubbleShape::Capsule;
		S.Radius = Radius;
		S.HalfHeight = HalfHeight;
		S.Material = Material;
		S.Tint = Tint;
		return Add(B, S, false);
	}

	int32 FBodyBuilder::Hull(const FVector& Center, int32 Mesh, float Radius, const FQuat& Rot, ERubbleMaterial Material, const FLinearColor& Tint, bool bAwake)
	{
		FBody B;
		B.Com = Center;
		B.Rot = Rot;
		B.HalfExtents = FVector(Radius * 0.8);
		B.Radius = Radius;
		B.bAwake = bAwake;
		B.Material = Material;
		FRubbleShapeVisual S;
		S.Kind = ERubbleShape::Hull;
		S.Mesh = Mesh;
		S.Material = Material;
		S.Tint = Tint;
		return Add(B, S, false);
	}

	int32 FBodyBuilder::ParkedBall(float Radius)
	{
		const int32 i = Sphere(FVector(0, 0, -100000.0), Radius, ERubbleMaterial::Steel, FLinearColor(0.55f, 0.56f, 0.58f), false);
		World.Bodies[i].bProjectile = true;
		World.Bodies[i].Mass = 200.0f;
		return i;
	}

	void FBodyBuilder::Fire(FWorld& World, int32 Ball, const FVector& Origin, const FVector& Direction, float Speed)
	{
		FBody& B = World.Bodies[Ball];
		B.Com = Origin;
		B.Rot = FQuat::Identity;
		B.Vel = FVector::ZeroVector;
		B.AngVel = FVector::ZeroVector;
		World.Wake(Ball, Direction.GetSafeNormal() * Speed);
		B.bWokeThisStep = true;
	}

	FLinearColor BrickTint(FRandomStream& Random)
	{
		const float v = Random.FRandRange(0.8f, 1.1f);
		return FLinearColor(0.55f * v, 0.22f * v, 0.15f * v);
	}

	FLinearColor StoneTint(FRandomStream& Random)
	{
		const float v = Random.FRandRange(0.75f, 1.05f);
		return FLinearColor(0.62f * v, 0.6f * v, 0.56f * v);
	}

	// ----- the scene -------------------------------------------------------------------------

	namespace
	{
		/** A group of bodies moved together by a script (ragdolls, chains, motors). */
		struct FKinematic
		{
			int32 Body = 0;
			int32 Group = 0;
			FTransform Local = FTransform::Identity; // in the group frame
			float Phase = 0.0f;
			float Rate = 0.0f;
			FVector Axis = FVector::UpVector;
			FVector Pivot = FVector::ZeroVector;
			float Amplitude = 0.0f;
		};

		struct FGroup
		{
			FVector Pos = FVector::ZeroVector;
			FVector Vel = FVector::ZeroVector;
			FQuat Rot = FQuat::Identity;
			FQuat Target = FQuat::Identity;
			bool bLanded = false;
			bool bResting = false;
			float Bottom = 0.0f;
		};

		class FRigidScene : public FScene
		{
		public:
			bool Build(const FRubbleSceneRequest& Request, FRubbleSceneAssets& Assets, FString& Error) override;
			void Step(float Dt, FRubbleFrame& Out) override;
			void Launch(const FVector& Origin, const FVector& Direction) override;

		private:
			FWorld World;
			TArray<int32> Balls;
			int32 NextBall = 0;
			float Time = 0.0f;
			float AutoShotAt = -1.0f;
			FVector AutoShotFrom = FVector::ZeroVector;
			FVector AutoShotAim = FVector::ZeroVector;
			TArray<FKinematic> Kinematic;
			TArray<FGroup> Groups;
			TArray<FRubbleEvent> Pending;
			enum class EAnim : uint8 { None, Ragdolls, Chains, Motors } Anim = EAnim::None;

			void Wall(FBodyBuilder& B, const FVector& Center, int32 Cols, int32 Rows, const FVector& Half, bool bRunning);
			void Ring(FBodyBuilder& B, float Radius, int32 Rows, const FVector& Half);
			// demo.html's builders (engine-webgpu/scene-builders.mjs), in this frame.
			static FVector WebPoint(double X, double Y, double Z);
			int32 WebBrick(FBodyBuilder& B, double X, double Y, double Z, double Yaw);
			void WebBall(FBodyBuilder& B, double X, double Y, double Z, double Radius, double Density);
			void WebRing(FBodyBuilder& B, double Radius, int32 Courses, int32 Tier, int32 Rows);
			void WebGables(FBodyBuilder& B, int32 Columns, int32 Rows, int32 Base);
			void Gables(FBodyBuilder& B, float Scale);
			void Pyramid(FBodyBuilder& B, int32 Levels, float Half, const FVector& At);
			void DropGrid(FBodyBuilder& B, int32 Nx, int32 Ny, int32 Nz, float Half, float Spacing, float Height, bool bSlanted);
			void Scatter(FBodyBuilder& B, int32 Count, float Area, bool bMixed);
			void Ragdolls(FBodyBuilder& B, int32 Count);
			void Chains(FBodyBuilder& B, int32 Chains, int32 Links);
			void Motors(FBodyBuilder& B, int32 Count);
		};

		void FRigidScene::Wall(FBodyBuilder& B, const FVector& Center, int32 Cols, int32 Rows, const FVector& Half, bool bRunning)
		{
			for (int32 r = 0; r < Rows; ++r)
			{
				const double Offset = bRunning && (r & 1) ? Half.Y : 0.0;
				for (int32 c = 0; c < Cols; ++c)
				{
					const FVector P = Center + FVector(0, (c - Cols * 0.5 + 0.5) * 2.0 * Half.Y + Offset, Half.Z + r * 2.0 * Half.Z);
					B.Box(P, Half, FQuat::Identity, ERubbleMaterial::Brick, BrickTint(World.Random), false);
				}
			}
		}

		void FRigidScene::Ring(FBodyBuilder& B, float Radius, int32 Rows, const FVector& Half)
		{
			const int32 PerRow = FMath::Max(8, FMath::FloorToInt32(UE_TWO_PI * Radius / (2.0 * Half.Y)));
			for (int32 r = 0; r < Rows; ++r)
			{
				for (int32 k = 0; k < PerRow; ++k)
				{
					const double A = (k + (r & 1) * 0.5) * UE_TWO_PI / PerRow;
					const FVector P(Radius * FMath::Cos(A), Radius * FMath::Sin(A), Half.Z + r * 2.0 * Half.Z);
					const FQuat Q(FVector::UpVector, A);
					B.Box(P, Half, Q, ERubbleMaterial::Brick, BrickTint(World.Random), false);
				}
			}
		}

		// demo.html's scenes are in the engine's frame (Y up, metres) with the ground's top at y = 0.5;
		// here Z is up, the ground at 0, centimetres: X = -z, Y = x, Z = y - 0.5.
		FVector FRigidScene::WebPoint(double X, double Y, double Z)
		{
			return FVector(-Z, X, Y - 0.5) * 100.0;
		}

		int32 FRigidScene::WebBrick(FBodyBuilder& B, double X, double Y, double Z, double Yaw)
		{
			// The page's 1 x 0.5 x 0.5 m brick of 0.25 kg, long along its x, yawed about up
			// (brickQuaternion: -yaw about +Y there, +yaw about +Z here).
			const int32 Body = B.Box(WebPoint(X, Y, Z), FVector(25, 50, 25), FQuat(FVector::UpVector, Yaw), ERubbleMaterial::Brick, BrickTint(World.Random), false, true);
			B.Assets.Bodies[Body].Mass = 0.25f;
			return Body;
		}

		void FRigidScene::WebBall(FBodyBuilder& B, double X, double Y, double Z, double Radius, double Density)
		{
			// Rolling along the page's +z at 30 m/s without slipping (rollingAngularVelocity).
			const int32 Body = B.Sphere(WebPoint(X, Y, Z), float(Radius * 100.0), ERubbleMaterial::Steel, FLinearColor(0.55f, 0.57f, 0.6f), true);
			FRubbleBodyVisual& V = B.Assets.Bodies[Body];
			V.Mass = float(Density * 4.0 / 3.0 * UE_DOUBLE_PI * Radius * Radius * Radius);
			V.InitialVelocity = FVector(-3000.0, 0.0, 0.0);
			V.InitialAngularVelocity = FVector(0.0, -30.0 / Radius, 0.0);
		}

		void FRigidScene::WebRing(FBodyBuilder& B, double Radius, int32 Courses, int32 Tier, int32 Rows)
		{
			for (int32 c = 0; c < Courses; ++c)
			{
				const int32 Depth = FMath::Max(1, Rows - c / Tier);
				for (int32 j = 0; j < Depth; ++j)
				{
					const double R = Radius + 0.25 + 0.52 * j;
					const int32 N = FMath::FloorToInt32(2.0 * UE_DOUBLE_PI * (R - 0.25) / 1.02);
					for (int32 i = 0; i < N; ++i)
					{
						const double A = (i + (c % 2) * 0.5) / N * 2.0 * UE_DOUBLE_PI;
						WebBrick(B, R * FMath::Cos(A), 0.75 + 0.5 * c, R * FMath::Sin(A), A + UE_DOUBLE_HALF_PI);
					}
				}
			}
			const double BallR = Courses / 5.0;
			WebBall(B, 0.0, 0.5 + BallR, -(Radius + 0.52 * Rows + BallR + 2.0), BallR, 10.0);
		}

		void FRigidScene::WebGables(FBodyBuilder& B, int32 Columns, int32 Rows, int32 Base)
		{
			const double PitchX = Base + 2, PitchY = 5;
			for (int32 cx = 0; cx < Columns; ++cx)
			{
				for (int32 cy = 0; cy < Rows; ++cy)
				{
					const double X0 = (cx - (Columns - 1) / 2.0) * PitchX;
					const double Z = (cy - (Rows - 1) / 2.0) * PitchY;
					for (int32 k = 0; k < Base; ++k)
					{
						const int32 M = Base - k;
						for (int32 i = 0; i < M; ++i)
						{
							WebBrick(B, X0 + (i - (M - 1) / 2.0) * 1.01, 0.75 + 0.5 * k, Z, 0.0);
						}
					}
				}
			}
			const double BallR = Base / 7.0;
			for (const double cx : { Columns / 2.0 - 2.0, Columns / 2.0 + 1.0 })
			{
				const double X = (cx - (Columns - 1) / 2.0) * PitchX;
				const double Z = -((Rows - 1) / 2.0) * PitchY - BallR - 12.0;
				WebBall(B, X, 0.5 + BallR, Z, BallR, 50.0);
			}
		}

		void FRigidScene::Gables(FBodyBuilder& B, float Scale)
		{
			// A house of bricks: two long walls, two gable ends with triangular tops.
			const FVector Half(12.0, 24.0, 8.0);
			const int32 Rows = FMath::RoundToInt32(24 * Scale), Long = FMath::RoundToInt32(30 * Scale), Short = FMath::RoundToInt32(18 * Scale);
			const double Lx = Long * 2.0 * Half.Y, Ly = Short * 2.0 * Half.Y;
			for (int32 Side = -1; Side <= 1; Side += 2)
			{
				for (int32 r = 0; r < Rows; ++r)
				{
					for (int32 c = 0; c < Long; ++c)
					{
						const double y = (c - Long * 0.5 + 0.5 + (r & 1) * 0.5) * 2.0 * Half.Y;
						B.Box(FVector(y, Side * Ly * 0.5, Half.Z + r * 2.0 * Half.Z), Half, FQuat(FVector::UpVector, UE_HALF_PI), ERubbleMaterial::Brick, BrickTint(World.Random), false);
					}
				}
				const int32 GableRows = Rows + Short / 2;
				for (int32 r = 0; r < GableRows; ++r)
				{
					const int32 Width = r < Rows ? Short : Short - 2 * (r - Rows) - 1;
					for (int32 c = 0; c < Width; ++c)
					{
						const double y = (c - Width * 0.5 + 0.5 + (r & 1) * 0.5) * 2.0 * Half.Y;
						B.Box(FVector(Side * Lx * 0.5, y, Half.Z + r * 2.0 * Half.Z), Half, FQuat::Identity, ERubbleMaterial::Brick, BrickTint(World.Random), false);
					}
				}
			}
		}

		void FRigidScene::Pyramid(FBodyBuilder& B, int32 Levels, float Half, const FVector& At)
		{
			for (int32 l = 0; l < Levels; ++l)
			{
				const int32 N = Levels - l;
				for (int32 k = 0; k < N; ++k)
				{
					const FVector P = At + FVector(0, (k - N * 0.5 + 0.5) * 2.02 * Half, Half + l * 2.0 * Half);
					B.Box(P, FVector(Half), FQuat::Identity, ERubbleMaterial::Timber, FLinearColor(0.75f, 0.55f, 0.35f) * World.Random.FRandRange(0.85f, 1.1f), false);
				}
			}
		}

		void FRigidScene::DropGrid(FBodyBuilder& B, int32 Nx, int32 Ny, int32 Nz, float Half, float Spacing, float Height, bool bSlanted)
		{
			for (int32 z = 0; z < Nz; ++z)
			{
				for (int32 x = 0; x < Nx; ++x)
				{
					for (int32 y = 0; y < Ny; ++y)
					{
						const FVector P((x - Nx * 0.5) * Spacing, (y - Ny * 0.5) * Spacing, Height + z * Spacing);
						const FQuat Q = bSlanted ? FQuat(FVector(1, 1, 0).GetSafeNormal(), 0.4) : FQuat::Identity;
						B.Box(P, FVector(Half), Q, ERubbleMaterial::Plastic, FLinearColor::MakeFromHSV8(uint8((x * 7 + y * 13 + z * 29) & 255), 150, 220), false, true);
					}
				}
			}
		}

		void FRigidScene::Scatter(FBodyBuilder& B, int32 Count, float Area, bool bMixed)
		{
			int32 Rock = INDEX_NONE;
			if (bMixed)
			{
				FRubbleMeshData Mesh;
				Mesh.Material = ERubbleMaterial::Concrete;
				AppendRock(Mesh, 30.0f, World.Random);
				Rock = B.Assets.Meshes.Add(MoveTemp(Mesh));
			}
			// A jittered lattice of 80 cm cells over the area, stacked upward: no two overlap.
			const int32 Side = FMath::Max(1, FMath::FloorToInt32(2.0f * Area / 80.0f));
			for (int32 i = 0; i < Count; ++i)
			{
				const int32 Layer = i / (Side * Side), InLayer = i % (Side * Side);
				const FVector P(-Area + 40.0 + (InLayer % Side) * 80.0 + World.Random.FRandRange(-8.f, 8.f),
					-Area + 40.0 + (InLayer / Side) * 80.0 + World.Random.FRandRange(-8.f, 8.f), 300.0 + Layer * 90.0 + World.Random.FRandRange(0.f, 30.f));
				const FQuat Q(World.Random.VRand(), World.Random.FRandRange(0.0f, UE_PI));
				const FLinearColor Tint = FLinearColor::MakeFromHSV8(uint8(World.Random.RandHelper(255)), 120, 230);
				switch (bMixed ? i % 4 : 0)
				{
				case 1: B.Sphere(P, 25.0f, ERubbleMaterial::Rubber, Tint, true); break;
				case 2: B.Capsule(P, 18.0f, 25.0f, Q, ERubbleMaterial::Plastic, Tint, true); break;
				case 3: B.Hull(P, Rock, 30.0f, Q, ERubbleMaterial::Concrete, StoneTint(World.Random), true); break;
				default: B.Box(P, FVector(25.0), Q, ERubbleMaterial::Plastic, Tint, false, true); break;
				}
			}
		}

		void FRigidScene::Ragdolls(FBodyBuilder& B, int32 Count)
		{
			Anim = EAnim::Ragdolls;
			// Head, torso, two arms, two legs: one group per ragdoll, dropped from a staggered height,
			// landing on its back.
			const int32 Side = FMath::CeilToInt32(FMath::Sqrt(float(Count)));
			for (int32 i = 0; i < Count; ++i)
			{
				FGroup G;
				G.Pos = FVector((i % Side - Side * 0.5) * 140.0, (i / Side - Side * 0.5) * 140.0, 400.0 + (i % 7) * 120.0 + (i / Side) * 15.0);
				G.Rot = FQuat(World.Random.VRand(), World.Random.FRandRange(0.0f, 0.6f));
				G.Target = FQuat(FVector(0, 1, 0), -UE_HALF_PI) * FQuat(FVector::UpVector, World.Random.FRandRange(0.0f, UE_TWO_PI));
				G.Bottom = 20.0f;
				const int32 Group = Groups.Add(G);
				const FLinearColor Cloth = FLinearColor::MakeFromHSV8(uint8(World.Random.RandHelper(255)), 140, 200);
				struct FPart { FVector At; float R; float H; FRotator Rot; bool bSphere; };
				const FPart Parts[] = {
					{ FVector(0, 0, 85), 12, 0, FRotator::ZeroRotator, true },
					{ FVector(0, 0, 45), 16, 18, FRotator::ZeroRotator, false },
					{ FVector(0, 28, 50), 6, 20, FRotator(0, 0, 25), false },
					{ FVector(0, -28, 50), 6, 20, FRotator(0, 0, -25), false },
					{ FVector(0, 10, 0), 8, 26, FRotator::ZeroRotator, false },
					{ FVector(0, -10, 0), 8, 26, FRotator::ZeroRotator, false },
				};
				for (const FPart& P : Parts)
				{
					const FTransform Local(P.Rot.Quaternion(), P.At);
					const FTransform World0 = Local * FTransform(G.Rot, G.Pos);
					const int32 Body = P.bSphere ? B.Sphere(World0.GetLocation(), P.R, ERubbleMaterial::Rubber, FLinearColor(0.85f, 0.65f, 0.5f), false)
												 : B.Capsule(World0.GetLocation(), P.R, P.H, World0.GetRotation(), ERubbleMaterial::Plastic, Cloth, false);
					World.Bodies[Body].bKinematic = true;
					FKinematic K;
					K.Body = Body;
					K.Group = Group;
					K.Local = Local;
					Kinematic.Add(K);
				}
			}
		}

		void FRigidScene::Chains(FBodyBuilder& B, int32 NumChains, int32 Links)
		{
			Anim = EAnim::Chains;
			const int32 Side = FMath::CeilToInt32(FMath::Sqrt(float(NumChains)));
			for (int32 c = 0; c < NumChains; ++c)
			{
				const FVector Pivot((c % Side - Side * 0.5) * 60.0, (c / Side - Side * 0.5) * 60.0, 60.0 * Links + 100.0);
				const float Phase = World.Random.FRandRange(0.0f, UE_TWO_PI);
				for (int32 l = 0; l < Links; ++l)
				{
					const int32 Body = B.Box(Pivot - FVector(0, 0, 30.0 + 60.0 * l), FVector(8, 8, 26), FQuat::Identity, ERubbleMaterial::Steel, FLinearColor(0.7f, 0.7f, 0.72f), false);
					World.Bodies[Body].bKinematic = true;
					FKinematic K;
					K.Body = Body;
					K.Pivot = Pivot;
					K.Phase = Phase - l * 0.18f;
					K.Rate = 1.6f;
					K.Amplitude = 0.5f;
					K.Local = FTransform(FVector(0, 0, -(30.0 + 60.0 * l)));
					K.Axis = FVector(1, 0, 0);
					Kinematic.Add(K);
				}
			}
		}

		void FRigidScene::Motors(FBodyBuilder& B, int32 Count)
		{
			Anim = EAnim::Motors;
			const int32 Side = FMath::CeilToInt32(FMath::Sqrt(float(Count)));
			for (int32 i = 0; i < Count; ++i)
			{
				const FVector P((i % Side - Side * 0.5) * 80.0, (i / Side - Side * 0.5) * 80.0, 40.0);
				const bool bMotor = (i % 4) == 0;
				const int32 Body = B.Box(P, bMotor ? FVector(35, 6, 6) : FVector(10), FQuat::Identity, bMotor ? ERubbleMaterial::Steel : ERubbleMaterial::Plastic,
					bMotor ? FLinearColor(0.8f, 0.6f, 0.1f) : FLinearColor(0.2f, 0.4f, 0.8f), false);
				World.Bodies[Body].bKinematic = true;
				FKinematic K;
				K.Body = Body;
				K.Pivot = P;
				K.Rate = bMotor ? World.Random.FRandRange(2.0f, 8.0f) : 0.0f;
				K.Amplitude = bMotor ? 0.0f : 15.0f;
				K.Phase = World.Random.FRandRange(0.0f, UE_TWO_PI);
				K.Axis = FVector::UpVector;
				Kinematic.Add(K);
			}
		}

		bool FRigidScene::Build(const FRubbleSceneRequest& R, FRubbleSceneAssets& A, FString& Error)
		{
			A.Stage = ERubbleStage::Yard;
			FBodyBuilder B{ World, A };
			const FString& Id = R.Id;
			auto Shot = [this](float At, const FVector& From, const FVector& Aim) {
				AutoShotAt = At;
				AutoShotFrom = From;
				AutoShotAim = Aim;
			};
			if (Id == TEXT("wall"))
			{
				Wall(B, FVector::ZeroVector, 50, 40, FVector(15, 30, 15), true);
				Shot(0.5f, FVector(-2500, 0, 300), FVector(0, 0, 250));
				A.CameraEye = FVector(-4200, -900, 1600);
				A.CameraTarget = FVector(0, 0, 500);
				A.GroundHalfExtentCm = 4800;
			}
			else if (Id == TEXT("ring") || Id == TEXT("ring110k"))
			{
				// demo.html's rings (scene-builders.mjs buildRing), its 4 iterations, friction 0.5,
				// no sleep (sleepTime 1e6), and its camera.
				const bool bBig = Id == TEXT("ring110k");
				if (bBig)
				{
					WebRing(B, 80.0f, 40, 4, 10);
				}
				else
				{
					WebRing(B, 12.0f, 20, 2, 10);
				}
				A.SolverIterations = 4;
				A.Friction = 0.5f;
				A.bSleep = false;
				A.CameraEye = bBig ? WebPoint(140, 60, 40) : WebPoint(32, 22, 14);
				A.CameraTarget = bBig ? WebPoint(0, 12, 0) : WebPoint(0, 5, 0);
				A.GroundHalfExtentCm = bBig ? 9000 : 2200;
			}
			else if (Id == TEXT("gables"))
			{
				// demo.html's gables (buildGables(4, 8, 20)): 3 iterations, friction 0.5.
				WebGables(B, 4, 8, 20);
				A.SolverIterations = 3;
				A.Friction = 0.5f;
				A.CameraEye = WebPoint(0, 28, -40);
				A.CameraTarget = WebPoint(0, 6, 0);
				A.GroundHalfExtentCm = 4400;
			}
			else if (Id == TEXT("gables506k"))
			{
				const bool bBig = true;
				Gables(B, bBig ? 8.7f : 1.0f);
				Shot(1.0f, FVector(-3000 * (bBig ? 4 : 1), 0, 300), FVector(-500, 0, 200));
				A.CameraEye = bBig ? FVector(-12000, -2000, 8000) : FVector(-4000, -600, 2800);
				A.CameraTarget = FVector(0, 0, bBig ? 1000 : 600);
				A.GroundHalfExtentCm = bBig ? 25600 : 4400;
			}
			else if (Id == TEXT("pyramid") || Id == TEXT("legacy-pyramid"))
			{
				const int32 Levels = Id == TEXT("pyramid") ? FMath::Clamp(FMath::RoundToInt32(R.Param(TEXT("levels"), 30)), 2, 50) : 16;
				Pyramid(B, Levels, 50.0f, FVector::ZeroVector);
				A.CameraEye = FVector(-2800, -2800, 2200);
				A.CameraTarget = FVector(0, 0, 600);
				A.GroundHalfExtentCm = 6400;
			}
			else if (Id == TEXT("pile"))
			{
				// 32k boxes in a mound, on a jittered lattice (no two overlap: a real solver starts
				// from it without exploding), the mound's shape the page's settled pile.
				int32 Made = 0;
				for (int32 z = 0; Made < 32000 && z < 40; ++z)
				{
					const double Radius = 2000.0 * (1.0 - z / 40.0);
					for (int32 x = -40; x <= 40 && Made < 32000; ++x)
					{
						for (int32 y = -40; y <= 40 && Made < 32000; ++y)
						{
							const FVector C(x * 56.0 + World.Random.FRandRange(-2.f, 2.f), y * 56.0 + World.Random.FRandRange(-2.f, 2.f), 26.0 + z * 51.0);
							if (C.Size2D() > Radius)
							{
								continue;
							}
							B.Box(C, FVector(25.0), FQuat(FVector::UpVector, World.Random.FRandRange(-0.3f, 0.3f)), ERubbleMaterial::Timber,
								FLinearColor(0.7f, 0.5f, 0.3f) * World.Random.FRandRange(0.7f, 1.1f), false);
							++Made;
						}
					}
				}
				A.CameraEye = FVector(-4800, -4800, 3600);
				A.CameraTarget = FVector(0, 0, 400);
				A.GroundHalfExtentCm = 9600;
			}
			else if (Id == TEXT("columns"))
			{
				for (int32 x = 0; x < 50; ++x)
				{
					for (int32 y = 0; y < 50; ++y)
					{
						for (int32 z = 0; z < 40; ++z)
						{
							B.Box(FVector((x - 25) * 150.0, (y - 25) * 150.0, 25.0 + z * 50.0), FVector(25.0), FQuat::Identity, ERubbleMaterial::Concrete, StoneTint(World.Random), false);
						}
					}
				}
				A.CameraEye = FVector(-9000, -9000, 4000);
				A.CameraTarget = FVector(0, 0, 600);
				A.GroundHalfExtentCm = 20000;
			}
			else if (Id == TEXT("joints") || Id == TEXT("hingechains10k"))
			{
				const bool bBig = Id == TEXT("hingechains10k");
				Chains(B, bBig ? 1000 : 12, bBig ? 10 : 8);
				A.CameraEye = bBig ? FVector(-3500, -3500, 1800) : FVector(-900, -900, 700);
				A.CameraTarget = FVector(0, 0, bBig ? 300 : 300);
				A.GroundHalfExtentCm = 5000;
			}
			else if (Id == TEXT("ragdolls") || Id == TEXT("ragdolls10k"))
			{
				Ragdolls(B, Id == TEXT("ragdolls") ? 40 : 1834);
				A.CameraEye = Id == TEXT("ragdolls") ? FVector(-1400, -1400, 900) : FVector(-6000, -6000, 3000);
				A.CameraTarget = FVector(0, 0, 100);
				A.GroundHalfExtentCm = 8000;
			}
			else if (Id == TEXT("motorfield10k"))
			{
				Motors(B, 10000);
				A.CameraEye = FVector(-4500, -4500, 2500);
				A.CameraTarget = FVector::ZeroVector;
				A.GroundHalfExtentCm = 6000;
			}
			else if (Id == TEXT("legacy-empty"))
			{
				A.GroundHalfExtentCm = 0.0f;
			}
			else if (Id == TEXT("legacy-ground"))
			{
			}
			else if (Id == TEXT("legacy-dynamic-friction") || Id == TEXT("legacy-static-friction"))
			{
				// A ramp with a row of boxes on it; they slide off at speeds by friction (scripted).
				const bool bStatic = Id == TEXT("legacy-static-friction");
				const FQuat Tilt(FVector(0, 1, 0), FMath::DegreesToRadians(bStatic ? 20.0 : 30.0));
				B.Box(FVector(0, 0, 300), FVector(800, 600, 20), Tilt, ERubbleMaterial::Concrete, FLinearColor(0.5f, 0.5f, 0.5f), true);
				for (int32 i = 0; i < 11; ++i)
				{
					const FVector Local(400.0, (i - 5) * 100.0, 50.0);
					const int32 Box = B.Box(FVector(0, 0, 300) + Tilt.RotateVector(Local), FVector(30), Tilt, ERubbleMaterial::Plastic,
						FLinearColor::MakeFromHSV8(uint8(i * 22), 160, 220), false);
					World.Bodies[Box].bKinematic = true;
					FKinematic K;
					K.Body = Box;
					K.Pivot = FVector(0, 0, 300);
					K.Local = FTransform(Tilt, Local);
					K.Rate = bStatic ? (i > 5 ? 0.0f : 60.0f * (5 - i)) : 40.0f * i;
					Kinematic.Add(K);
				}
				Anim = EAnim::Motors;
				A.CameraEye = FVector(-1500, -1800, 900);
				A.CameraTarget = FVector(0, 0, 300);
			}
			else if (Id == TEXT("legacy-stack"))
			{
				for (int32 i = 0; i < 10; ++i)
				{
					B.Box(FVector(0, 0, 50 + i * 100.0), FVector(50), FQuat::Identity, ERubbleMaterial::Timber, FLinearColor(0.7f, 0.5f, 0.3f), false);
				}
				A.CameraEye = FVector(-1500, -600, 600);
				A.CameraTarget = FVector(0, 0, 450);
			}
			else if (Id == TEXT("legacy-stack-ratio"))
			{
				B.Box(FVector(0, 0, 400), FVector(400), FQuat::Identity, ERubbleMaterial::Concrete, FLinearColor(0.6f, 0.6f, 0.6f), false);
				B.Box(FVector(0, 0, 850), FVector(50), FQuat::Identity, ERubbleMaterial::Timber, FLinearColor(0.7f, 0.5f, 0.3f), false);
				A.CameraEye = FVector(-2000, -900, 1100);
				A.CameraTarget = FVector(0, 0, 500);
			}
			else if (Id == TEXT("legacy-scatter") || Id == TEXT("legacy-scatter-boxes"))
			{
				const bool bMixed = Id == TEXT("legacy-scatter");
				Scatter(B, bMixed ? 3000 : 4000, 1500.0f, bMixed);
				A.CameraEye = FVector(-3000, -3000, 1800);
				A.CameraTarget = FVector(0, 0, 200);
			}
			else if (Id == TEXT("legacy-grid-boxes"))
			{
				DropGrid(B, 12, 12, 10, 25.0f, 70.0f, 200.0f, false);
				A.CameraEye = FVector(-1800, -1800, 1200);
				A.CameraTarget = FVector(0, 0, 200);
			}
			else if (Id == TEXT("legacy-slanted-grid"))
			{
				DropGrid(B, 16, 20, 10, 25.0f, 70.0f, 200.0f, true);
				A.CameraEye = FVector(-2000, -2000, 1300);
				A.CameraTarget = FVector(0, 0, 200);
			}
			else if (Id == TEXT("legacy-grid10k"))
			{
				DropGrid(B, 25, 20, 20, 25.0f, 70.0f, 200.0f, false);
				A.CameraEye = FVector(-2800, -2800, 1800);
				A.CameraTarget = FVector(0, 0, 400);
			}
			else if (Id == TEXT("legacy-grid100k"))
			{
				DropGrid(B, 50, 50, 40, 25.0f, 70.0f, 200.0f, false);
				A.CameraEye = FVector(-6000, -6000, 4000);
				A.CameraTarget = FVector(0, 0, 800);
				A.GroundHalfExtentCm = 12000;
			}
			else
			{
				Error = FString::Printf(TEXT("no rigid scene '%s'"), *Id);
				return false;
			}
			// demo.html runs these with reuseContacts (flags: 1).
			A.bReuseContacts = Id == TEXT("ring") || Id == TEXT("ring110k") || Id == TEXT("gables") || Id == TEXT("gables506k")
				|| Id == TEXT("pyramid") || Id == TEXT("pile") || Id == TEXT("columns");
			World.BuildSupports();
			for (int32 i = 0; i < 16; ++i)
			{
				Balls.Add(B.ParkedBall(Id == TEXT("wall") || Id.StartsWith(TEXT("ring")) || Id.StartsWith(TEXT("gables")) ? 60.0f : 40.0f));
			}
			return true;
		}

		void FRigidScene::Launch(const FVector& Origin, const FVector& Direction)
		{
			if (Balls.Num() == 0)
			{
				return;
			}
			FBodyBuilder::Fire(World, Balls[NextBall], Origin, Direction, 4000.0f);
			NextBall = (NextBall + 1) % Balls.Num();
		}

		void FRigidScene::Step(float Dt, FRubbleFrame& Out)
		{
			Time += Dt;
			if (AutoShotAt >= 0.0f && Time >= AutoShotAt)
			{
				AutoShotAt = -1.0f;
				Launch(AutoShotFrom, AutoShotAim - AutoShotFrom + FVector(0, 0, 0.5 * GravityCm * FMath::Square((AutoShotAim - AutoShotFrom).Size() / 4000.0)));
			}
			if (!World.OnProjectileHit)
			{
				World.OnProjectileHit = [this](int32 Projectile, int32 Hit, const FVector& Point) {
					World.Blast(Point, World.BlastRadius, World.BlastSpeed * 0.8f, World.Bodies[Projectile].Vel.GetSafeNormal());
					FRubbleEvent E;
					E.Type = ERubbleEventType::Impact;
					E.Position = Point;
					E.Normal = -World.Bodies[Projectile].Vel.GetSafeNormal();
					E.Magnitude = float(World.Bodies[Projectile].Vel.Size());
					E.MaterialA = ERubbleMaterial::Steel;
					E.MaterialB = World.Bodies[Hit].Material;
					Pending.Add(E);
					return true;
				};
			}
			World.Step(Dt, Out);
			Out.Events.Append(MoveTemp(Pending));
			Pending.Reset();

			// Scripted bodies.
			for (FGroup& G : Groups)
			{
				if (G.bResting)
				{
					continue;
				}
				G.Vel.Z -= GravityCm * Dt;
				G.Pos += G.Vel * Dt;
				if (G.Pos.Z < G.Bottom)
				{
					G.Pos.Z = G.Bottom;
					G.Vel = G.Vel.Z < -300.0 ? FVector(G.Vel.X * 0.5, G.Vel.Y * 0.5, -G.Vel.Z * 0.2) : FVector::ZeroVector;
					G.bLanded = true;
				}
				if (G.bLanded)
				{
					G.Rot = FQuat::Slerp(G.Rot, G.Target, FMath::Min(1.0f, 4.0f * Dt));
					G.bResting = G.Vel.IsNearlyZero() && G.Rot.Equals(G.Target, 1e-3);
				}
			}
			for (const FKinematic& K : Kinematic)
			{
				FBody& Body = World.Bodies[K.Body];
				FTransform T;
				bool bAsleep = false;
				switch (Anim)
				{
				case EAnim::Ragdolls:
				{
					const FGroup& G = Groups[K.Group];
					T = K.Local * FTransform(G.Rot, G.Pos);
					bAsleep = G.bResting;
					if (bAsleep && !Body.bAwake)
					{
						continue;
					}
					break;
				}
				case EAnim::Chains:
				{
					const float Angle = K.Amplitude * FMath::Sin(K.Rate * Time + K.Phase);
					T = K.Local * FTransform(FQuat(K.Axis, Angle), K.Pivot);
					break;
				}
				default:
					if (K.Amplitude > 0.0f)
					{
						T = FTransform(FQuat::Identity, K.Pivot + FVector(0, 0, K.Amplitude * FMath::Abs(FMath::Sin(3.0f * Time + K.Phase))));
					}
					else if (!K.Local.Equals(FTransform::Identity))
					{
						// A box sliding down a ramp at its friction's speed, until it leaves the ramp.
						const double Slide = FMath::Min(1600.0, K.Rate * Time);
						T = FTransform(FVector(-Slide, 0, 0)) * K.Local * FTransform(K.Pivot);
						if (K.Rate == 0.0f)
						{
							continue;
						}
					}
					else
					{
						T = FTransform(FQuat(K.Axis, K.Rate * Time + K.Phase), K.Pivot);
					}
					break;
				}
				Body.SetOrigin(T);
				Body.bAwake = !bAsleep;
				World.Report(K.Body, Out, bAsleep);
			}
		}
	}

	TUniquePtr<FScene> MakeRigidScene()
	{
		return TUniquePtr<FScene>(new FRigidScene());
	}
}

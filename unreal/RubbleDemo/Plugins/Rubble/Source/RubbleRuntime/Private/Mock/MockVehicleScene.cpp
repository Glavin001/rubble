// vehicle.html's scenes as mock layouts: kinematic bicycle-model cars over ramps and ledges, wheel
// visuals (spin, steer, suspension), AI traffic on loops, parts that break off on a hard hit, and
// crates and bricks the car wakes. The ids and titles are the page's.

#include "Mock/MockScene.h"

namespace RubbleMock
{
	namespace
	{
		FRubbleDemoDesc Vehicle(const TCHAR* Id, const TCHAR* Title, TArray<FRubbleParamDesc> Params = {})
		{
			FRubbleDemoDesc D;
			D.Family = ERubbleFamily::Vehicles;
			D.Id = Id;
			D.Title = Title;
			D.Params = MoveTemp(Params);
			D.StepHz = 60.0;
			return D;
		}
	}

	TArray<FRubbleDemoDesc> VehicleCatalog()
	{
		return {
			Vehicle(TEXT("playground"), TEXT("Playground: ramps, stacks, wall")),
			Vehicle(TEXT("destruction"), TEXT("Destruction yard: sever parts")),
			Vehicle(TEXT("traffic"), TEXT("Traffic: many AI vehicles"), { { TEXT("vehicles"), TEXT("Vehicles"), 48, 1, 60 } }),
			Vehicle(TEXT("slab"), TEXT("Lab: car on a slab sliding on ice")),
			Vehicle(TEXT("seesaw"), TEXT("Lab: seesaw plank")),
			Vehicle(TEXT("carry"), TEXT("Lab: kart riding on a truck")),
			Vehicle(TEXT("sidewall"), TEXT("Lab: crates beside the tyres")),
			Vehicle(TEXT("bridge"), TEXT("Lab: plank bridge with breakable bolts")),
			Vehicle(TEXT("tanklab"), TEXT("Lab: tank over rubble, logs and a ledge")),
			Vehicle(TEXT("buggies"), TEXT("Dune buggies: jumps, walls, cannon (parts break off under load)")),
		};
	}

	namespace
	{
		struct FPreset
		{
			FVector Half;
			float WheelRadius;
			float WheelWidth;
			float Wheelbase;
			float Track;
			int32 Axles;
			float Ride; // chassis centre above the hubs
			float Travel;
			float MaxSpeed;
			float Accel;
			FLinearColor Paint;
		};

		FPreset PresetOf(ERubbleVehicleStyle S)
		{
			switch (S)
			{
			case ERubbleVehicleStyle::Sports: return { FVector(220, 92, 32), 33, 26, 265, 162, 2, 22, 12, 7000, 900, FLinearColor(0.85f, 0.75f, 0.05f) };
			case ERubbleVehicleStyle::MonsterTruck: return { FVector(240, 110, 60), 80, 60, 300, 230, 2, 70, 45, 4500, 700, FLinearColor(0.1f, 0.5f, 0.15f) };
			case ERubbleVehicleStyle::SixWheelTruck: return { FVector(380, 120, 90), 50, 36, 560, 200, 3, 60, 20, 3200, 350, FLinearColor(0.85f, 0.45f, 0.1f) };
			case ERubbleVehicleStyle::Kart: return { FVector(80, 55, 15), 14, 14, 105, 100, 2, 6, 3, 4000, 900, FLinearColor(0.1f, 0.3f, 0.85f) };
			case ERubbleVehicleStyle::Tank: return { FVector(330, 160, 70), 30, 50, 500, 260, 7, 45, 15, 1800, 300, FLinearColor(0.32f, 0.36f, 0.22f) };
			case ERubbleVehicleStyle::Motorcycle: return { FVector(105, 18, 40), 32, 14, 145, 0, 2, 35, 15, 6000, 1000, FLinearColor(0.08f, 0.08f, 0.08f) };
			case ERubbleVehicleStyle::DuneBuggy: return { FVector(170, 85, 40), 40, 30, 240, 170, 2, 45, 35, 5000, 900, FLinearColor(0.9f, 0.35f, 0.05f) };
			default: return { FVector(230, 90, 42), 33, 22, 270, 156, 2, 30, 14, 5000, 600, FLinearColor(0.6f, 0.05f, 0.04f) };
			}
		}

		struct FCar
		{
			int32 Visual = 0;
			int32 Body = 0;
			ERubbleVehicleStyle Style = ERubbleVehicleStyle::Sedan;
			FPreset P;
			FVector Pos = FVector::ZeroVector; // chassis centre
			float Yaw = 0.0f;
			float Speed = 0.0f;
			float Steer = 0.0f;
			float VelZ = 0.0f;
			float Pitch = 0.0f;
			float Roll = 0.0f;
			bool bAir = false;
			bool bAI = false;
			FVector2D LoopCenter = FVector2D::ZeroVector;
			float LoopRadius = 0.0f;
			float LoopSpeed = 0.0f;
			TArray<float> Spin;
			TArray<int32> Parts; // breakable parts riding on the chassis (destruction, buggies)
			TArray<FTransform> PartLocal;
			TArray<FName> PartNames;
			bool bMoving = true;
		};

		struct FObstacle
		{
			FVector Center;
			FVector Half;
			float Yaw = 0.0f;
			float Pitch = 0.0f; // ramps
			bool bRamp = false;
		};

		class FVehicleScene : public FScene
		{
		public:
			bool Build(const FRubbleSceneRequest& Request, FRubbleSceneAssets& Assets, FString& Error) override;
			void Step(float Dt, FRubbleFrame& Out) override;
			void SetVehicleInput(const FRubbleVehicleInput& InInput) override { Input = InInput; }
			void Launch(const FVector& Origin, const FVector& Direction) override;

		private:
			FWorld World;
			FRubbleSceneAssets* A = nullptr;
			TArray<FCar> Cars;
			TArray<FObstacle> Obstacles;
			TArray<int32> Balls;
			TArray<int32> Bridge;
			int32 Seesaw = INDEX_NONE;
			int32 Slab = INDEX_NONE;
			int32 NextBall = 0;
			float Time = 0.0f;
			float CannonAt = -1.0f;
			float OnBridge = 0.0f;
			FRubbleVehicleInput Input;
			TArray<FRubbleEvent> Pending;

			int32 AddCar(FBodyBuilder& B, ERubbleVehicleStyle Style, const FVector2D& At, float Yaw, bool bPlayer);
			void AddRamp(FBodyBuilder& B, const FVector& Center, const FVector& Half, float Yaw, float PitchDeg);
			void AddWall(FBodyBuilder& B, const FVector& Center, const FVector& Half, float Yaw);
			void Crates(FBodyBuilder& B, const FVector& At, int32 Levels);
			float GroundAt(const FVector2D& P, float Below) const;
			void StepCar(FCar& C, float Dt, FRubbleFrame& Out);
		};

		int32 FVehicleScene::AddCar(FBodyBuilder& B, ERubbleVehicleStyle Style, const FVector2D& At, float Yaw, bool bPlayer)
		{
			FCar C;
			C.Style = Style;
			C.P = PresetOf(Style);
			C.Yaw = Yaw;
			C.Pos = FVector(At, C.P.WheelRadius + C.P.Ride);
			// The chassis: a body the vehicle renderer draws (no primitive shape).
			FBody Body;
			Body.Com = C.Pos;
			Body.Rot = FQuat(FVector::UpVector, Yaw);
			Body.bKinematic = true;
			Body.HalfExtents = C.P.Half;
			Body.Radius = float(C.P.Half.Size());
			C.Body = World.Add(Body);
			FRubbleBodyVisual BV;
			BV.Id = FWorld::IdOf(C.Body);
			BV.Initial = Body.Origin();
			BV.FirstShape = A->Shapes.Num();
			BV.NumShapes = 0;
			A->Bodies.Add(BV);

			FRubbleVehicleVisual V;
			V.Vehicle = A->Vehicles.Num();
			V.Chassis = BV.Id;
			V.Style = Style;
			V.ChassisHalfExtents = FVector3f(C.P.Half);
			V.Paint = bPlayer ? C.P.Paint : FLinearColor::MakeFromHSV8(uint8(World.Random.RandHelper(255)), 180, 200);
			V.bPlayer = bPlayer;
			const int32 PerSide = C.P.Axles;
			for (int32 a = 0; a < PerSide; ++a)
			{
				const float x = PerSide == 1 ? 0.0f : C.P.Wheelbase * (0.5f - float(a) / (PerSide - 1));
				const int32 Sides = Style == ERubbleVehicleStyle::Motorcycle ? 1 : 2;
				for (int32 s = 0; s < Sides; ++s)
				{
					FRubbleWheelDef W;
					W.LocalHub = FVector3f(x, Sides == 1 ? 0.0f : (s == 0 ? -0.5f : 0.5f) * C.P.Track, -C.P.Ride);
					W.Radius = C.P.WheelRadius;
					W.Width = C.P.WheelWidth;
					V.Wheels.Add(W);
				}
			}
			C.Visual = A->Vehicles.Add(V);
			C.Spin.Init(0.0f, V.Wheels.Num());
			C.bAI = !bPlayer;
			return Cars.Add(C);
		}

		void FVehicleScene::AddRamp(FBodyBuilder& B, const FVector& Center, const FVector& Half, float Yaw, float PitchDeg)
		{
			const FQuat Q = FQuat(FVector::UpVector, Yaw) * FQuat(FVector(0, 1, 0), -FMath::DegreesToRadians(PitchDeg));
			B.Box(Center, Half, Q, ERubbleMaterial::Concrete, FLinearColor(0.55f, 0.55f, 0.53f), true);
			Obstacles.Add({ Center, Half, Yaw, FMath::DegreesToRadians(PitchDeg), true });
		}

		void FVehicleScene::AddWall(FBodyBuilder& B, const FVector& Center, const FVector& Half, float Yaw)
		{
			B.Box(Center, Half, FQuat(FVector::UpVector, Yaw), ERubbleMaterial::Concrete, FLinearColor(0.62f, 0.6f, 0.57f), true);
			Obstacles.Add({ Center, Half, Yaw, 0.0f, false });
		}

		void FVehicleScene::Crates(FBodyBuilder& B, const FVector& At, int32 Levels)
		{
			for (int32 l = 0; l < Levels; ++l)
			{
				for (int32 i = 0; i < Levels - l; ++i)
				{
					for (int32 j = 0; j < Levels - l; ++j)
					{
						const FVector P = At + FVector((i - (Levels - l) * 0.5 + 0.5) * 82.0, (j - (Levels - l) * 0.5 + 0.5) * 82.0, 40.0 + l * 80.0);
						B.Box(P, FVector(40.0), FQuat::Identity, ERubbleMaterial::Timber, FLinearColor(0.72f, 0.52f, 0.3f) * World.Random.FRandRange(0.8f, 1.1f), false);
					}
				}
			}
		}

		float FVehicleScene::GroundAt(const FVector2D& P, float Below) const
		{
			float Z = 0.0f;
			for (const FObstacle& O : Obstacles)
			{
				const FVector2D D = P - FVector2D(O.Center);
				const float c = FMath::Cos(-O.Yaw), s = FMath::Sin(-O.Yaw);
				const FVector2D L(D.X * c - D.Y * s, D.X * s + D.Y * c);
				if (O.bRamp)
				{
					// Top face of a slab pitched about its local Y (nose up toward +X).
					const float th = O.Pitch;
					const float xl = (L.X + O.Half.Z * FMath::Sin(th)) / FMath::Cos(th);
					if (FMath::Abs(xl) <= O.Half.X && FMath::Abs(L.Y) <= O.Half.Y)
					{
						Z = FMath::Max(Z, float(O.Center.Z + xl * FMath::Sin(th) + O.Half.Z * FMath::Cos(th)));
					}
				}
				else if (FMath::Abs(L.X) <= O.Half.X && FMath::Abs(L.Y) <= O.Half.Y)
				{
					const float Top = float(O.Center.Z + O.Half.Z);
					if (Top <= Below + 45.0f)
					{
						Z = FMath::Max(Z, Top); // a ledge the wheels climb
					}
				}
			}
			return Z;
		}

		bool FVehicleScene::Build(const FRubbleSceneRequest& R, FRubbleSceneAssets& Assets, FString& Error)
		{
			A = &Assets;
			Assets.Stage = ERubbleStage::Yard;
			Assets.GroundHalfExtentCm = 15000.0f;
			FBodyBuilder B{ World, Assets };
			const FString& Id = R.Id;
			ERubbleVehicleStyle Player = ERubbleVehicleStyle::Sedan;
			if (Id == TEXT("carry")) Player = ERubbleVehicleStyle::MonsterTruck;
			if (Id == TEXT("tanklab")) Player = ERubbleVehicleStyle::Tank;
			if (Id == TEXT("buggies")) Player = ERubbleVehicleStyle::DuneBuggy;

			if (Id == TEXT("playground") || Id == TEXT("destruction"))
			{
				AddRamp(B, FVector(2500, 0, 60), FVector(500, 300, 30), 0.0f, 12.0f);
				AddRamp(B, FVector(-2000, 2500, 40), FVector(400, 250, 30), UE_HALF_PI, 8.0f);
				AddRamp(B, FVector(0, -3500, 80), FVector(700, 400, 30), -0.6f, 15.0f);
				Crates(B, FVector(1200, 1800, 0), 4);
				Crates(B, FVector(-1500, -1200, 0), 5);
				// A brick wall to drive through.
				for (int32 r = 0; r < 10; ++r)
				{
					for (int32 c = 0; c < 20; ++c)
					{
						B.Box(FVector(4500, (c - 10 + 0.5 * (r & 1)) * 60.0, 15.0 + r * 30.0), FVector(15, 30, 15), FQuat::Identity, ERubbleMaterial::Brick, BrickTint(World.Random), false);
					}
				}
				AddWall(B, FVector(-5000, 0, 150), FVector(50, 2000, 150), 0.0f);
			}
			else if (Id == TEXT("traffic"))
			{
				const int32 N = FMath::Clamp(FMath::RoundToInt32(R.Param(TEXT("vehicles"), 48)), 1, 60);
				static const ERubbleVehicleStyle Styles[] = { ERubbleVehicleStyle::Sedan, ERubbleVehicleStyle::Sports, ERubbleVehicleStyle::SixWheelTruck, ERubbleVehicleStyle::Kart, ERubbleVehicleStyle::MonsterTruck };
				for (int32 i = 0; i < N; ++i)
				{
					const int32 Lane = i % 4;
					const float Radius = 2500.0f + Lane * 600.0f;
					const float Angle = UE_TWO_PI * float(i / 4) / FMath::Max(1, (N + 3) / 4) + Lane * 0.3f;
					const int32 c = AddCar(B, Styles[i % 5], FVector2D(Radius * FMath::Cos(Angle), Radius * FMath::Sin(Angle)), Angle + UE_HALF_PI, false);
					Cars[c].LoopRadius = Radius;
					Cars[c].LoopSpeed = World.Random.FRandRange(900.0f, 1700.0f);
				}
			}
			else if (Id == TEXT("slab"))
			{
				Slab = B.Box(FVector(0, 0, 10), FVector(600, 400, 10), FQuat::Identity, ERubbleMaterial::Steel, FLinearColor(0.7f, 0.8f, 0.9f), false);
				World.Bodies[Slab].bKinematic = true;
			}
			else if (Id == TEXT("seesaw"))
			{
				B.Box(FVector(0, 0, 50), FVector(60, 300, 50), FQuat::Identity, ERubbleMaterial::Concrete, FLinearColor(0.5f, 0.5f, 0.5f), true);
				Seesaw = B.Box(FVector(0, 0, 112), FVector(900, 250, 12), FQuat::Identity, ERubbleMaterial::Timber, FLinearColor(0.65f, 0.45f, 0.25f), false);
				World.Bodies[Seesaw].bKinematic = true;
			}
			else if (Id == TEXT("carry"))
			{
				const int32 Kart = AddCar(B, ERubbleVehicleStyle::Kart, FVector2D(0, 0), 0.0f, false);
				Cars[Kart].bAI = false;
				Cars[Kart].LoopRadius = -1.0f; // rides on the player's truck (see Step)
			}
			else if (Id == TEXT("sidewall"))
			{
				for (int32 i = 0; i < 12; ++i)
				{
					B.Box(FVector(-600 + i * 100.0, 140, 40), FVector(40), FQuat::Identity, ERubbleMaterial::Timber, FLinearColor(0.72f, 0.52f, 0.3f), false);
					B.Box(FVector(-600 + i * 100.0, -140, 40), FVector(40), FQuat::Identity, ERubbleMaterial::Timber, FLinearColor(0.72f, 0.52f, 0.3f), false);
				}
			}
			else if (Id == TEXT("bridge"))
			{
				AddRamp(B, FVector(-1500, 0, 100), FVector(520, 220, 20), 0.0f, 11.0f);
				AddRamp(B, FVector(1500, 0, 100), FVector(520, 220, 20), UE_PI, 11.0f);
				for (int32 i = 0; i < 20; ++i)
				{
					const FVector C(-950 + i * 100.0, 0, 190);
					Bridge.Add(B.Box(C, FVector(45, 220, 6), FQuat::Identity, ERubbleMaterial::Timber, FLinearColor(0.6f, 0.42f, 0.24f) * World.Random.FRandRange(0.85f, 1.1f), false));
					Obstacles.Add({ C, FVector(50, 220, 6), 0.0f, 0.0f, false });
				}
			}
			else if (Id == TEXT("tanklab"))
			{
				FRubbleMeshData Rock;
				AppendRock(Rock, 40.0f, World.Random);
				const int32 RockMesh = Assets.Meshes.Add(MoveTemp(Rock));
				for (int32 i = 0; i < 60; ++i)
				{
					B.Hull(FVector(World.Random.FRandRange(500, 1800), World.Random.FRandRange(-500, 500), 35), RockMesh, 40.0f, FQuat(World.Random.VRand(), World.Random.FRandRange(0.f, 3.f)), ERubbleMaterial::Concrete, StoneTint(World.Random));
				}
				for (int32 i = 0; i < 8; ++i)
				{
					B.Capsule(FVector(2300 + i * 90.0, 0, 25), 25.0f, 280.0f, FQuat(FVector(1, 0, 0), UE_HALF_PI), ERubbleMaterial::Timber, FLinearColor(0.45f, 0.3f, 0.18f));
				}
				AddWall(B, FVector(3600, 0, 20), FVector(300, 600, 20), 0.0f); // the ledge: 40 cm up
			}
			else if (Id == TEXT("buggies"))
			{
				for (int32 i = 0; i < 4; ++i)
				{
					AddRamp(B, FVector(2000 + i * 2200, (i & 1) ? 600 : -600, 70), FVector(450, 280, 30), 0.0f, 16.0f);
				}
				AddWall(B, FVector(11000, 0, 200), FVector(60, 1500, 200), 0.0f);
				CannonAt = 3.0f;
			}
			else
			{
				Error = FString::Printf(TEXT("no vehicle scene '%s'"), *Id);
				return false;
			}

			const int32 PlayerCar = AddCar(B, Player, FVector2D(-400.0, 0.0), 0.0f, true);
			Swap(Cars[PlayerCar], Cars[0]);
			if (Id == TEXT("destruction") || Id == TEXT("buggies"))
			{
				// Breakable parts: front bumper, bonnet, two doors, riding on the chassis until a hard hit.
				FCar& C = Cars[0];
				const FVector H = C.P.Half;
				struct FPart { const TCHAR* Name; FVector At; FVector Half; };
				const FPart Parts[] = {
					{ TEXT("bumper_front"), FVector(H.X + 8, 0, -H.Z * 0.4), FVector(8, H.Y * 0.95, 12) },
					{ TEXT("bonnet"), FVector(H.X * 0.6, 0, H.Z + 2), FVector(H.X * 0.35, H.Y * 0.9, 3) },
					{ TEXT("door_l"), FVector(0, -H.Y - 3, 0), FVector(H.X * 0.3, 3, H.Z * 0.8) },
					{ TEXT("door_r"), FVector(0, H.Y + 3, 0), FVector(H.X * 0.3, 3, H.Z * 0.8) },
				};
				const FLinearColor Paint = Assets.Vehicles[C.Visual].Paint;
				for (const FPart& P : Parts)
				{
					const int32 Part = B.Box(C.Pos + P.At, P.Half, FQuat::Identity, ERubbleMaterial::CarPaint, Paint, false);
					World.Bodies[Part].bKinematic = true;
					C.Parts.Add(Part);
					C.PartLocal.Add(FTransform(P.At));
					C.PartNames.Add(P.Name);
				}
			}
			World.BuildSupports();
			for (int32 i = 0; i < 8; ++i)
			{
				Balls.Add(B.ParkedBall(25.0f));
			}
			Assets.CameraTarget = Cars[0].Pos;
			Assets.CameraEye = Cars[0].Pos + FVector(-900, 0, 350);
			return true;
		}

		void FVehicleScene::Launch(const FVector& Origin, const FVector& Direction)
		{
			if (Balls.Num())
			{
				FBodyBuilder::Fire(World, Balls[NextBall], Origin, Direction, 3500.0f);
				NextBall = (NextBall + 1) % Balls.Num();
			}
		}

		void FVehicleScene::StepCar(FCar& C, float Dt, FRubbleFrame& Out)
		{
			const bool bPlayer = &C == &Cars[0];
			float Throttle = 0.0f, Steer = 0.0f, Brake = 0.0f;
			if (bPlayer)
			{
				Throttle = Input.Throttle;
				Steer = Input.Steer;
				Brake = Input.Brake + (Input.bHandbrake ? 1.0f : 0.0f);
			}
			else if (C.bAI && C.LoopRadius > 0.0f)
			{
				// Hold the loop: steer toward a point ahead on the circle.
				const FVector2D P(C.Pos);
				const float Angle = FMath::Atan2(float(P.Y), float(P.X)) + 0.25f;
				const FVector2D Aim(C.LoopRadius * FMath::Cos(Angle), C.LoopRadius * FMath::Sin(Angle));
				const float Want = FMath::Atan2(float(Aim.Y - P.Y), float(Aim.X - P.X));
				Steer = FMath::Clamp(FMath::FindDeltaAngleRadians(C.Yaw, Want) * 2.0f, -1.0f, 1.0f);
				Throttle = C.Speed < C.LoopSpeed ? 0.6f : 0.0f;
			}
			else if (C.LoopRadius < 0.0f)
			{
				// carry: the kart rides on the truck's bed.
				const FCar& Truck = Cars[0];
				const FQuat Q = FQuat(FVector::UpVector, Truck.Yaw) * FQuat(FVector(0, 1, 0), -Truck.Pitch) * FQuat(FVector(1, 0, 0), Truck.Roll);
				C.Pos = Truck.Pos + Q.RotateVector(FVector(-60, 0, Truck.P.Half.Z + C.P.Ride + C.P.WheelRadius));
				C.Yaw = Truck.Yaw;
				C.Pitch = Truck.Pitch;
				C.Roll = Truck.Roll;
				C.Speed = Truck.Speed;
			}

			const bool bTank = C.Style == ERubbleVehicleStyle::Tank;
			C.Steer = FMath::FInterpTo(C.Steer, Steer, Dt, 8.0f);
			const float Drag = 0.25f + 0.0004f * FMath::Abs(C.Speed);
			C.Speed += (Throttle * C.P.Accel - Drag * C.Speed) * Dt;
			if (Brake > 0.0f)
			{
				C.Speed = FMath::FInterpTo(C.Speed, 0.0f, Dt, 3.0f * Brake);
			}
			C.Speed = FMath::Clamp(C.Speed, -0.4f * C.P.MaxSpeed, C.P.MaxSpeed);
			if (C.LoopRadius >= 0.0f)
			{
				const float YawRate = bTank ? C.Steer * 0.9f : C.Speed / C.P.Wheelbase * FMath::Tan(C.Steer * 0.6f);
				C.Yaw += YawRate * Dt;
				C.Pos += FVector(FMath::Cos(C.Yaw), FMath::Sin(C.Yaw), 0.0) * C.Speed * Dt;
			}

			// Ground under the axles.
			const FVector Fwd(FMath::Cos(C.Yaw), FMath::Sin(C.Yaw), 0.0);
			const FVector Side(-Fwd.Y, Fwd.X, 0.0);
			const float HalfBase = C.P.Wheelbase * 0.5f, HalfTrack = FMath::Max(10.0f, C.P.Track * 0.5f);
			const float Below = float(C.Pos.Z - C.P.Ride - C.P.WheelRadius);
			const float Gf = GroundAt(FVector2D(C.Pos + Fwd * HalfBase), Below);
			const float Gb = GroundAt(FVector2D(C.Pos - Fwd * HalfBase), Below);
			const float Gl = GroundAt(FVector2D(C.Pos - Side * HalfTrack), Below);
			const float Gr = GroundAt(FVector2D(C.Pos + Side * HalfTrack), Below);
			const float Target = 0.25f * (Gf + Gb + Gl + Gr) + C.P.WheelRadius + C.P.Ride;
			if (C.LoopRadius >= 0.0f)
			{
				if (C.bAir || C.Pos.Z > Target + 6.0f)
				{
					C.bAir = true;
					C.VelZ -= GravityCm * Dt;
					C.Pos.Z += C.VelZ * Dt;
					if (C.Pos.Z <= Target)
					{
						if (C.VelZ < -400.0f)
						{
							FRubbleEvent E;
							E.Type = ERubbleEventType::Impact;
							E.Position = FVector(C.Pos.X, C.Pos.Y, Target - C.P.Ride - C.P.WheelRadius);
							E.Magnitude = -C.VelZ;
							E.MaterialA = ERubbleMaterial::Rubber;
							Out.Events.Add(E);
						}
						C.Pos.Z = Target;
						C.VelZ = 0.0f;
						C.bAir = false;
					}
				}
				else
				{
					C.VelZ = (Target - float(C.Pos.Z)) / Dt;
					C.Pos.Z = Target;
				}
				if (!C.bAir)
				{
					C.Pitch = FMath::FInterpTo(C.Pitch, FMath::Atan2(Gf - Gb, 2.0f * HalfBase), Dt, 10.0f);
					const float Lean = C.Style == ERubbleVehicleStyle::Motorcycle ? -C.Steer * FMath::Clamp(C.Speed / 2500.0f, 0.0f, 0.6f) : 0.0f;
					C.Roll = FMath::FInterpTo(C.Roll, FMath::Atan2(Gr - Gl, 2.0f * HalfTrack) + Lean, Dt, 10.0f);
				}
			}

			// Walls and other cars stop it.
			for (const FObstacle& O : Obstacles)
			{
				if (O.bRamp || O.Center.Z + O.Half.Z <= Below + 45.0f)
				{
					continue;
				}
				const FVector2D D = FVector2D(C.Pos) - FVector2D(O.Center);
				const float c = FMath::Cos(-O.Yaw), s = FMath::Sin(-O.Yaw);
				const FVector2D L(D.X * c - D.Y * s, D.X * s + D.Y * c);
				const FVector2D Reach(O.Half.X + C.P.Half.X * 0.9f, O.Half.Y + C.P.Half.Y);
				if (FMath::Abs(L.X) < Reach.X && FMath::Abs(L.Y) < Reach.Y)
				{
					const FVector2D Pen(Reach.X - FMath::Abs(L.X), Reach.Y - FMath::Abs(L.Y));
					FVector2D Push = Pen.X < Pen.Y ? FVector2D(FMath::Sign(L.X) * Pen.X, 0) : FVector2D(0, FMath::Sign(L.Y) * Pen.Y);
					const FVector2D W(Push.X * FMath::Cos(O.Yaw) - Push.Y * FMath::Sin(O.Yaw), Push.X * FMath::Sin(O.Yaw) + Push.Y * FMath::Cos(O.Yaw));
					C.Pos += FVector(W, 0.0);
					const float HitSpeed = FMath::Abs(C.Speed);
					C.Speed *= -0.25f;
					if (HitSpeed > 300.0f)
					{
						FRubbleEvent E;
						E.Type = ERubbleEventType::Impact;
						E.Position = C.Pos + Fwd * C.P.Half.X * FMath::Sign(-C.Speed);
						E.Normal = FVector(W.GetSafeNormal(), 0.0);
						E.Magnitude = HitSpeed;
						E.MaterialA = ERubbleMaterial::CarPaint;
						E.MaterialB = ERubbleMaterial::Concrete;
						Out.Events.Add(E);
					}
					if (HitSpeed > 900.0f && C.Parts.Num())
					{
						// The part nearest the hit breaks off with the car's speed.
						const int32 k = 0;
						const int32 Part = C.Parts[k];
						FRubbleEvent E;
						E.Type = ERubbleEventType::JointBroke;
						E.Position = World.Bodies[Part].Com;
						E.Magnitude = HitSpeed * 1500.0f;
						E.Part = C.PartNames[k];
						Out.Events.Add(E);
						World.Bodies[Part].bKinematic = false;
						World.Wake(Part, Fwd * HitSpeed * 0.4f + FVector(0, 0, 350) + World.Random.VRand() * 150.0f);
						World.Bodies[Part].AngVel = World.Random.VRand() * 5.0;
						C.Parts.RemoveAt(k);
						C.PartLocal.RemoveAt(k);
						C.PartNames.RemoveAt(k);
					}
				}
			}

			// Crates and bricks in the way are knocked aside.
			const float Reach = C.P.Half.X;
			for (int32 i = 0; i < World.Bodies.Num(); ++i)
			{
				FBody& O = World.Bodies[i];
				if (O.bStatic || O.bKinematic || O.bProjectile || !O.bAlive || i == C.Body)
				{
					continue;
				}
				const FVector D = O.Com - C.Pos;
				if (D.SizeSquared2D() < FMath::Square(Reach + O.Radius) && O.Com.Z < C.Pos.Z + C.P.Half.Z + 30.0 && FMath::Abs(C.Speed) > 100.0f)
				{
					const FVector Away = (D.GetSafeNormal2D() + Fwd * FMath::Sign(C.Speed)).GetSafeNormal();
					World.Wake(i, Away * FMath::Abs(C.Speed) * 0.9f + FVector(0, 0, 150.0 + 0.15 * FMath::Abs(C.Speed)) - O.Vel * 0.5);
					O.AngVel += World.Random.VRand() * 4.0;
					C.Speed *= 0.985f;
				}
			}

			// Report the chassis, the parts on it, and the wheels.
			const FQuat Q = FQuat(FVector::UpVector, C.Yaw) * FQuat(FVector(0, 1, 0), -C.Pitch) * FQuat(FVector(1, 0, 0), C.Roll);
			FBody& Body = World.Bodies[C.Body];
			const bool bMoved = !Body.bReportedOnce || !Body.Com.Equals(C.Pos, 0.05) || !Body.Rot.Equals(Q, 1e-5);
			Body.Com = C.Pos;
			Body.Rot = Q;
			if (bMoved || C.bMoving)
			{
				World.Report(C.Body, Out, !bMoved);
				for (int32 k = 0; k < C.Parts.Num(); ++k)
				{
					World.Bodies[C.Parts[k]].SetOrigin(C.PartLocal[k] * FTransform(Q, C.Pos));
					World.Report(C.Parts[k], Out, !bMoved);
				}
			}
			C.bMoving = bMoved;

			const FTransform Chassis(Q, C.Pos);
			const FRubbleVehicleVisual& V = A->Vehicles[C.Visual];
			for (int32 w = 0; w < V.Wheels.Num(); ++w)
			{
				const FRubbleWheelDef& Def = V.Wheels[w];
				const FVector Rest = Chassis.TransformPosition(FVector(Def.LocalHub) - FVector(0, 0, C.P.Travel));
				const float G = GroundAt(FVector2D(Rest), float(Rest.Z - Def.Radius));
				const float Lift = FMath::Max(0.0f, G + Def.Radius - float(Rest.Z));
				const float Compression = C.bAir ? 0.0f : FMath::Clamp(Lift / FMath::Max(1.0f, 2.0f * C.P.Travel), 0.0f, 1.0f);
				const bool bFront = Def.LocalHub.X > 1.0f;
				const float SteerRad = bTank ? 0.0f : (bFront ? C.Steer * 0.6f : 0.0f);
				const float WheelSide = Def.LocalHub.Y < 0.0f ? -1.0f : 1.0f;
				const float SpinRate = bTank ? (C.Speed + WheelSide * C.Steer * 300.0f) / Def.Radius : C.Speed / Def.Radius;
				C.Spin[w] = FMath::Fmod(C.Spin[w] + SpinRate * Dt, UE_TWO_PI);
				FRubbleWheelVisual WV;
				WV.Vehicle = C.Visual;
				WV.Wheel = w;
				WV.Hub = FTransform(Q * FQuat(FVector::UpVector, SteerRad), Rest + FVector(0, 0, FMath::Min(Lift, 2.0f * C.P.Travel)));
				WV.SpinRad = C.Spin[w];
				WV.SteerRad = SteerRad;
				WV.Compression = Compression;
				WV.bOnGround = !C.bAir;
				Out.Wheels.Add(WV);
			}
		}

		void FVehicleScene::Step(float Dt, FRubbleFrame& Out)
		{
			Time += Dt;
			if (!World.OnProjectileHit)
			{
				World.OnProjectileHit = [this](int32 Projectile, int32 Hit, const FVector& Point) {
					World.Blast(Point, 150.0f, 700.0f, World.Bodies[Projectile].Vel.GetSafeNormal());
					FRubbleEvent E;
					E.Type = ERubbleEventType::Impact;
					E.Position = Point;
					E.Magnitude = float(World.Bodies[Projectile].Vel.Size());
					E.MaterialA = ERubbleMaterial::Steel;
					E.MaterialB = World.Bodies[Hit].Material;
					Pending.Add(E);
					return true;
				};
			}
			if (CannonAt >= 0.0f && Time >= CannonAt)
			{
				// The buggies' cannon: a ball toward the player every 2.5 s.
				CannonAt = Time + 2.5f;
				const FVector Target = Cars[0].Pos;
				const FVector From = Target + FVector(3000, 1500, 200);
				const double T = (Target - From).Size() / 3500.0;
				Launch(From, (Target - From) + FVector(0, 0, 0.5 * GravityCm * T * T));
			}
			if (Slab != INDEX_NONE)
			{
				World.Bodies[Slab].Com = FVector(400.0 * FMath::Sin(0.4f * Time), 0, 10);
				World.Report(Slab, Out);
			}
			if (Seesaw != INDEX_NONE)
			{
				// Tilts toward whichever end the car is on.
				const double Want = FMath::Clamp(Cars[0].Pos.X / 900.0, -1.0, 1.0) * 0.18;
				World.Bodies[Seesaw].Rot = FQuat::Slerp(World.Bodies[Seesaw].Rot, FQuat(FVector(0, 1, 0), Want), FMath::Min(1.0, 3.0 * Dt));
				World.Report(Seesaw, Out);
			}
			if (Bridge.Num())
			{
				const bool bOn = FMath::Abs(Cars[0].Pos.X) < 1000.0 && FMath::Abs(Cars[0].Pos.Y) < 220.0;
				OnBridge = bOn ? OnBridge + Dt : 0.0f;
				if (OnBridge > 1.5f)
				{
					// The bolts give: the planks under the car fall, one break event each.
					for (int32 k = Bridge.Num() - 1; k >= 0; --k)
					{
						FBody& Plank = World.Bodies[Bridge[k]];
						if (FMath::Abs(Plank.Com.X - Cars[0].Pos.X) < 250.0)
						{
							FRubbleEvent E;
							E.Type = ERubbleEventType::JointBroke;
							E.Position = Plank.Com;
							E.Magnitude = 20000.0f;
							E.Part = TEXT("bolt");
							Out.Events.Add(E);
							World.Wake(Bridge[k], FVector(0, 0, -50));
							Obstacles.RemoveAll([&Plank](const FObstacle& O) { return !O.bRamp && O.Center.Equals(Plank.Com, 1.0); });
							Bridge.RemoveAt(k);
						}
					}
				}
			}
			for (FCar& C : Cars)
			{
				StepCar(C, Dt, Out);
			}
			World.Step(Dt, Out);
			Out.Events.Append(MoveTemp(Pending));
			Pending.Reset();
		}
	}

	TUniquePtr<FScene> MakeVehicleScene()
	{
		return TUniquePtr<FScene>(new FVehicleScene());
	}
}

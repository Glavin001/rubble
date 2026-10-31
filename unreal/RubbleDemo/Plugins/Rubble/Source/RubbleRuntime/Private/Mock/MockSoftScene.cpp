// softbody.html's scenes as mock layouts. A soft body is a render mesh with a plastic offset per
// vertex: obstacles (walls, poles, rams, other cars, the ground) push vertices out, the offset
// spreads to neighbours as a crumple zone, parts tear off as pieces, and a rigid core rides on
// sprung wheels. Stand-in motion for tuning the renderer, not the node-beam solver. The ids,
// titles and parameters are the page's (?speed=56&overlap=0.25 ...).

#include "Mock/MockScene.h"

namespace RubbleMock
{
	namespace
	{
		FRubbleDemoDesc Soft(const TCHAR* Id, const TCHAR* Title, TArray<FRubbleParamDesc> Params = {})
		{
			FRubbleDemoDesc D;
			D.Family = ERubbleFamily::Soft;
			D.Id = Id;
			D.Title = Title;
			D.Params = MoveTemp(Params);
			D.StepHz = 240.0; // soft scenes step at 240 Hz (engine_softbody::soft_scene_params)
			return D;
		}
		FRubbleParamDesc Speed(float Default) { return { TEXT("speed"), TEXT("Speed (km/h)"), Default, 5, 160 }; }
	}

	TArray<FRubbleDemoDesc> SoftCatalog()
	{
		return {
			Soft(TEXT("yard"), TEXT("Crash yard (drive)")),
			Soft(TEXT("wall"), TEXT("Crash test: rigid wall, 54 km/h"), { Speed(54), { TEXT("overlap"), TEXT("Overlap"), 1, 0.1f, 1 }, { TEXT("angle"), TEXT("Wall angle (°)"), 0, -60, 60 } }),
			Soft(TEXT("pole"), TEXT("Crash test: 20 cm pole, 40 km/h"), { Speed(40), { TEXT("offset"), TEXT("Offset (m)"), 0, -0.9f, 0.9f } }),
			Soft(TEXT("tbone"), TEXT("Crash test: 1,500 kg side impact, 40 km/h"), { Speed(40), { TEXT("at"), TEXT("At (m forward)"), 0, -2, 2 }, { TEXT("mass"), TEXT("Mass (kg)"), 1500, 500, 20000 }, { TEXT("carSpeed"), TEXT("Car speed (km/h)"), 0, 0, 100 } }),
			Soft(TEXT("headon"), TEXT("Crash test: head-on, 2 × 29 km/h"), { Speed(29), { TEXT("overlap"), TEXT("Overlap"), 1, 0.1f, 1 } }),
			Soft(TEXT("sidepole"), TEXT("Crash test: side pole, 32 km/h"), { Speed(32) }),
			Soft(TEXT("rollover"), TEXT("Rollover: sliding sideways into a curb"), { Speed(60) }),
			Soft(TEXT("ramp"), TEXT("Corkscrew ramp (one side)"), { Speed(90) }),
			Soft(TEXT("drop"), TEXT("Nose-first drop"), { { TEXT("height"), TEXT("Height (m)"), 8, 1, 30 }, { TEXT("pitch"), TEXT("Pitch (°)"), 50, 0, 90 } }),
			Soft(TEXT("crates"), TEXT("Lattice crates drop")),
			Soft(TEXT("import"), TEXT("Crash test: your .truck / JBeam into a wall, 40 km/h"), { Speed(40) }),
			Soft(TEXT("traffic"), TEXT("Traffic derby: self-driving soft cars (?cars=10…100)"), { { TEXT("cars"), TEXT("Cars"), 20, 1, 100 } }),
			Soft(TEXT("limbo"), TEXT("Limbo bar: low clearance shears the roof off"), { Speed(80), { TEXT("bar"), TEXT("Bar height (m)"), 1.2f, 0.6f, 2 } }),
			Soft(TEXT("sandwich"), TEXT("Sandwich: 18 t truck shunts the car into a parked truck"), { Speed(60) }),
			Soft(TEXT("crusher"), TEXT("Crusher: 10 t block dropped on the roof"), { { TEXT("mass"), TEXT("Mass (kg)"), 10000, 1000, 50000 }, { TEXT("height"), TEXT("Height (m)"), 4, 1, 20 } }),
		};
	}

	namespace
	{
		constexpr float KmhToCms = 100000.0f / 3600.0f;

		/** An oriented box: walls, poles, rams, other cars' cores. */
		struct FObb
		{
			FVector Center = FVector::ZeroVector;
			FQuat Rot = FQuat::Identity;
			FVector Half = FVector(50.0);
			FVector Vel = FVector::ZeroVector;
			float Mass = 0.0f; // 0: immovable
			int32 Body = INDEX_NONE; // its rigid body in the world, if drawn as one
			bool bGravity = false;
		};

		enum EPart : uint8 { PartBody, PartBumperFront, PartBumperRear, PartBonnet, PartBoot, PartDoorFL, PartDoorFR, PartDoorRL, PartDoorRR, PartRoof, PartCount };
		const TCHAR* PartName(uint8 P)
		{
			static const TCHAR* Names[] = { TEXT("body"), TEXT("bumper_front"), TEXT("bumper_rear"), TEXT("bonnet"), TEXT("boot"), TEXT("door_fl"), TEXT("door_fr"), TEXT("door_rl"), TEXT("door_rr"), TEXT("roof") };
			return Names[P];
		}

		struct FPiece
		{
			FVector Pos;
			FQuat Rot;
			FVector Vel;
			FVector AngVel;
			bool bResting = false;
		};

		struct FSoft
		{
			int32 Visual = 0;
			int32 Mesh = 0;
			TArray<FVector3f> Rest;
			TArray<FVector3f> Offset;
			TArray<uint8> Part;
			TArray<float> Noise;
			TArray<TArray<int32>> Adj;
			TArray<uint32> Indices;
			TArray<float> RestEdge; // 3 per triangle
			TArray<FPiece> Pieces;  // per part, once detached
			TArray<int8> PieceOf;   // per part: index into Pieces, or -1
			TArray<float> PartCrush;
			FVector3f Half = FVector3f(225, 90, 72);
			FVector Pos = FVector::ZeroVector;
			FQuat Rot = FQuat::Identity;
			FVector Vel = FVector::ZeroVector;
			FVector AngVel = FVector::ZeroVector;
			float Mass = 1300.0f;
			float Inertia = 1.0f;
			bool bWheels = true;
			TArray<FRubbleWheelDef> Wheels;
			TArray<float> Spin;
			float Steer = 0.0f;
			float Drive = 0.0f; // target forward speed (cm/s) for scripted and AI cars
			bool bPlayer = false;
			bool bAI = false;
			FVector2D Loop = FVector2D::ZeroVector;
			float LoopRadius = 0.0f;
			float Still = 0.0f;
			/** Sliding sideways (sidepole, rollover): no tyre grip until it stops. */
			bool bNoGrip = false;
			bool bAsleep = false;
			bool bChanged = true;
			int32 BreaksThisStep = 0;
		};

		class FSoftScene : public FScene
		{
		public:
			bool Build(const FRubbleSceneRequest& Request, FRubbleSceneAssets& Assets, FString& Error) override;
			void Step(float Dt, FRubbleFrame& Out) override;
			void SetVehicleInput(const FRubbleVehicleInput& InInput) override { Input = InInput; }
			void SideRam() override;
			void Launch(const FVector& Origin, const FVector& Direction) override;

		private:
			FWorld World;
			FRubbleSceneAssets* A = nullptr;
			TArray<FSoft> Softs;
			TArray<FObb> Obstacles; // static and moving boxes
			TArray<int32> Balls;
			int32 NextBall = 0;
			int32 Ram = INDEX_NONE; // obstacle index of the side ram
			FRubbleVehicleInput Input;
			float Time = 0.0f;

			int32 AddCar(FBodyBuilder& B, const FVector& Pos, float Yaw, float Pitch, const FLinearColor& Paint, bool bPlayer);
			int32 AddCrate(FBodyBuilder& B, const FVector& Pos, float HalfSize);
			int32 AddObstacle(FBodyBuilder& B, const FVector& Center, const FVector& Half, const FQuat& Rot, ERubbleMaterial Material, const FLinearColor& Tint, float Mass = 0.0f, const FVector& Vel = FVector::ZeroVector);
			void FinishSoft(FSoft& S);
			float GroundAt(const FVector& P) const;
			void StepSoft(int32 Index, float Dt, FRubbleFrame& Out);
		};

		// ----- meshes ---------------------------------------------------------------------------

		/** A car shell: a superellipse cross-section swept along the length, cabin on top. */
		void CarMesh(FRubbleMeshData& M, TArray<uint8>& Parts, const FVector3f& Half)
		{
			const int32 Nu = 56, Nv = 44;
			const float L = Half.X, W = Half.Y;
			const float Bottom = 22.0f;
			auto Top = [L](float x) {
				// Bonnet 92 cm, windscreen up to the roof at 142 cm, boot 100 cm.
				const float u = (x + L) / (2.0f * L);
				const float Bonnet = 92.0f, Roof = 142.0f, Boot = 100.0f;
				if (u > 0.72f) return FMath::Lerp(Bonnet, Bonnet - 18.0f, FMath::Square((u - 0.72f) / 0.28f));
				if (u > 0.56f) return FMath::Lerp(Roof, Bonnet, FMath::SmoothStep(0.56f, 0.72f, u));
				if (u > 0.28f) return Roof;
				if (u > 0.16f) return FMath::Lerp(Boot, Roof, FMath::SmoothStep(0.16f, 0.28f, u));
				return FMath::Lerp(Boot - 8.0f, Boot, u / 0.16f);
			};
			auto Section = [&](float x, float v) {
				const float zTop = Top(x);
				const float Mid = 0.5f * (zTop + Bottom), HalfH = 0.5f * (zTop - Bottom);
				const float c = FMath::Cos(v), s = FMath::Sin(v);
				const float Pw = 0.35f; // superellipse exponent 2/n, n ~ 5.7: boxy, rounded corners
				float y = W * FMath::Sign(c) * FMath::Pow(FMath::Abs(c), Pw);
				const float z = Mid + HalfH * FMath::Sign(s) * FMath::Pow(FMath::Abs(s), Pw);
				// Tumblehome: the greenhouse narrows above the beltline (96 cm).
				y *= FMath::Lerp(1.0f, 0.74f, FMath::SmoothStep(96.0f, 142.0f, z));
				// Nose and tail round off in plan view.
				const float u = FMath::Abs(x) / L;
				y *= FMath::Lerp(1.0f, 0.82f, FMath::SmoothStep(0.86f, 1.0f, u));
				return FVector3f(x, y, z);
			};
			for (int32 i = 0; i < Nu; ++i)
			{
				const float x = FMath::Lerp(-L, L, float(i) / (Nu - 1));
				for (int32 j = 0; j < Nv; ++j)
				{
					const float v = UE_TWO_PI * float(j) / Nv;
					M.Positions.Add(Section(x, v));
					M.UVs.Add(FVector2f(float(i) / (Nu - 1) * 4.5f, float(j) / Nv * 4.0f));
				}
			}
			for (int32 i = 0; i + 1 < Nu; ++i)
			{
				for (int32 j = 0; j < Nv; ++j)
				{
					const uint32 a = i * Nv + j, b = i * Nv + (j + 1) % Nv, c = (i + 1) * Nv + j, d = (i + 1) * Nv + (j + 1) % Nv;
					M.Indices.Append({ a, b, d, a, d, c });
				}
			}
			// Caps: a fan around each end's centre.
			for (int32 End = 0; End < 2; ++End)
			{
				const int32 i = End == 0 ? 0 : Nu - 1;
				FVector3f C = FVector3f::ZeroVector;
				for (int32 j = 0; j < Nv; ++j)
				{
					C += M.Positions[i * Nv + j];
				}
				const uint32 Center = M.Positions.Add(C / Nv);
				M.UVs.Add(FVector2f(0.5f, 0.5f));
				for (int32 j = 0; j < Nv; ++j)
				{
					M.Indices.Append({ Center, uint32(i * Nv + j), uint32(i * Nv + (j + 1) % Nv) });
				}
			}
			// Outward winding (Unreal: (c - a) x (b - a) points out), fixed per triangle against
			// the direction from the core: the shell is star-shaped about it.
			const FVector3f Core(0, 0, 70);
			for (int32 t = 0; t < M.Indices.Num(); t += 3)
			{
				const FVector3f& Pa = M.Positions[M.Indices[t]];
				const FVector3f& Pb = M.Positions[M.Indices[t + 1]];
				const FVector3f& Pc = M.Positions[M.Indices[t + 2]];
				const FVector3f N = FVector3f::CrossProduct(Pc - Pa, Pb - Pa);
				if (FVector3f::DotProduct(N, (Pa + Pb + Pc) / 3.0f - Core) < 0.0f)
				{
					Swap(M.Indices[t + 1], M.Indices[t + 2]);
				}
			}
			// Parts by region.
			Parts.SetNum(M.Positions.Num());
			for (int32 v = 0; v < M.Positions.Num(); ++v)
			{
				const FVector3f& P = M.Positions[v];
				const float u = (P.X + L) / (2.0f * L);
				const bool bSide = FMath::Abs(P.Y) > 0.8f * W && P.Z > 38.0f && P.Z < 100.0f;
				uint8 Part = PartBody;
				if (u > 0.95f && P.Z < 62.0f) Part = PartBumperFront;
				else if (u < 0.05f && P.Z < 62.0f) Part = PartBumperRear;
				else if (u > 0.74f && u < 0.95f && P.Z > 80.0f) Part = PartBonnet;
				else if (u < 0.16f && P.Z > 88.0f) Part = PartBoot;
				else if (u > 0.3f && u < 0.54f && P.Z > 138.0f) Part = PartRoof;
				else if (bSide && u > 0.45f && u < 0.66f) Part = P.Y < 0 ? PartDoorFL : PartDoorFR;
				else if (bSide && u > 0.25f && u < 0.45f) Part = P.Y < 0 ? PartDoorRL : PartDoorRR;
				Parts[v] = Part;
			}
			M.Material = ERubbleMaterial::CarPaint;
			ComputeNormals(M.Positions, M.Indices, M.Normals);
		}

		/** A lattice crate: a subdivided box shell. */
		void CrateMesh(FRubbleMeshData& M, float H)
		{
			const int32 N = 8;
			static const FVector3f Axes[6][3] = {
				{ { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } }, { { -1, 0, 0 }, { 0, 0, 1 }, { 0, 1, 0 } },
				{ { 0, 1, 0 }, { 0, 0, 1 }, { 1, 0, 0 } }, { { 0, -1, 0 }, { 1, 0, 0 }, { 0, 0, 1 } },
				{ { 0, 0, 1 }, { 1, 0, 0 }, { 0, 1, 0 } }, { { 0, 0, -1 }, { 0, 1, 0 }, { 1, 0, 0 } },
			};
			for (const auto& F : Axes)
			{
				const uint32 Base = M.Positions.Num();
				for (int32 i = 0; i <= N; ++i)
				{
					for (int32 j = 0; j <= N; ++j)
					{
						const float s = 2.0f * i / N - 1.0f, t = 2.0f * j / N - 1.0f;
						M.Positions.Add((F[0] + F[1] * s + F[2] * t) * H);
						M.UVs.Add(FVector2f(float(i) / N, float(j) / N) * (H / 50.0f));
					}
				}
				for (int32 i = 0; i < N; ++i)
				{
					for (int32 j = 0; j < N; ++j)
					{
						const uint32 a = Base + i * (N + 1) + j, b = a + 1, c = a + (N + 1), d = c + 1;
						M.Indices.Append({ a, b, d, a, d, c });
					}
				}
			}
			for (int32 t = 0; t < M.Indices.Num(); t += 3)
			{
				const FVector3f& Pa = M.Positions[M.Indices[t]];
				const FVector3f& Pb = M.Positions[M.Indices[t + 1]];
				const FVector3f& Pc = M.Positions[M.Indices[t + 2]];
				if (FVector3f::DotProduct(FVector3f::CrossProduct(Pc - Pa, Pb - Pa), Pa + Pb + Pc) < 0.0f)
				{
					Swap(M.Indices[t + 1], M.Indices[t + 2]);
				}
			}
			M.Material = ERubbleMaterial::Timber;
			ComputeNormals(M.Positions, M.Indices, M.Normals);
		}

		/** Does the vertical line through P hit the box from above? The highest hit's Z. */
		bool TopOf(const FObb& O, const FVector& P, float& OutZ)
		{
			const FVector Origin = O.Rot.UnrotateVector(FVector(P.X, P.Y, 1e6) - O.Center);
			const FVector Dir = O.Rot.UnrotateVector(FVector(0, 0, -1));
			double TMin = -1e12, TMax = 1e12;
			for (int32 a = 0; a < 3; ++a)
			{
				if (FMath::Abs(Dir[a]) < 1e-9)
				{
					if (FMath::Abs(Origin[a]) > O.Half[a]) return false;
					continue;
				}
				double t1 = (-O.Half[a] - Origin[a]) / Dir[a], t2 = (O.Half[a] - Origin[a]) / Dir[a];
				if (t1 > t2) Swap(t1, t2);
				TMin = FMath::Max(TMin, t1);
				TMax = FMath::Min(TMax, t2);
				if (TMin > TMax) return false;
			}
			OutZ = float(1e6 - TMin);
			return true;
		}

		// ----- building -------------------------------------------------------------------------

		void FSoftScene::FinishSoft(FSoft& S)
		{
			const FRubbleMeshData& M = A->Meshes[S.Mesh];
			S.Rest = M.Positions;
			S.Indices = M.Indices;
			S.Offset.Init(FVector3f::ZeroVector, S.Rest.Num());
			S.Adj.SetNum(S.Rest.Num());
			S.Noise.SetNum(S.Rest.Num());
			for (float& n : S.Noise)
			{
				n = World.Random.FRandRange(-1.0f, 1.0f);
			}
			for (int32 t = 0; t < S.Indices.Num(); t += 3)
			{
				for (int32 k = 0; k < 3; ++k)
				{
					const int32 a = S.Indices[t + k], b = S.Indices[t + (k + 1) % 3];
					S.Adj[a].AddUnique(b);
					S.Adj[b].AddUnique(a);
					S.RestEdge.Add(FVector3f::Dist(S.Rest[a], S.Rest[b]));
				}
			}
			if (S.Part.Num() != S.Rest.Num())
			{
				S.Part.Init(PartBody, S.Rest.Num());
			}
			S.PieceOf.Init(-1, PartCount);
			S.PartCrush.Init(0.0f, PartCount);
			const float L = S.Half.X * 2.0f, W = S.Half.Y * 2.0f;
			S.Inertia = S.Mass * (L * L + W * W) / 12.0f;
			S.Spin.Init(0.0f, S.Wheels.Num());
		}

		int32 FSoftScene::AddCar(FBodyBuilder& B, const FVector& Pos, float Yaw, float Pitch, const FLinearColor& Paint, bool bPlayer)
		{
			FSoft S;
			FRubbleMeshData Mesh;
			CarMesh(Mesh, S.Part, S.Half);
			S.Mesh = A->Meshes.Add(MoveTemp(Mesh));
			S.Pos = Pos;
			S.Rot = FQuat(FVector::UpVector, Yaw) * FQuat(FVector(0, 1, 0), -Pitch);
			S.bPlayer = bPlayer;
			for (int32 a = 0; a < 2; ++a)
			{
				for (int32 s = 0; s < 2; ++s)
				{
					FRubbleWheelDef W;
					W.LocalHub = FVector3f(a == 0 ? 138.0f : -135.0f, s == 0 ? -79.0f : 79.0f, 33.0f); // tyre faces flush with the sides
					W.Radius = 33.0f;
					W.Width = 22.0f;
					S.Wheels.Add(W);
				}
			}
			FRubbleSoftVisual V;
			V.Soft = A->SoftBodies.Num();
			V.Mesh = S.Mesh;
			V.Paint = Paint;
			V.Wheels = S.Wheels;
			V.bPlayer = bPlayer;
			S.Visual = A->SoftBodies.Add(V);
			FinishSoft(S);
			return Softs.Add(MoveTemp(S));
		}

		int32 FSoftScene::AddCrate(FBodyBuilder& B, const FVector& Pos, float HalfSize)
		{
			FSoft S;
			FRubbleMeshData Mesh;
			CrateMesh(Mesh, HalfSize);
			S.Mesh = A->Meshes.Add(MoveTemp(Mesh));
			S.Pos = Pos;
			S.Rot = FQuat(World.Random.VRand(), World.Random.FRandRange(0.0f, 0.5f));
			S.Half = FVector3f(HalfSize);
			S.Mass = 120.0f;
			S.bWheels = false;
			FRubbleSoftVisual V;
			V.Soft = A->SoftBodies.Num();
			V.Mesh = S.Mesh;
			V.Paint = FLinearColor(0.7f, 0.5f, 0.28f);
			S.Visual = A->SoftBodies.Add(V);
			FinishSoft(S);
			return Softs.Add(MoveTemp(S));
		}

		int32 FSoftScene::AddObstacle(FBodyBuilder& B, const FVector& Center, const FVector& Half, const FQuat& Rot, ERubbleMaterial Material, const FLinearColor& Tint, float Mass, const FVector& Vel)
		{
			FObb O;
			O.Center = Center;
			O.Half = Half;
			O.Rot = Rot;
			O.Mass = Mass;
			O.Vel = Vel;
			O.Body = B.Box(Center, Half, Rot, Material, Tint, Mass == 0.0f);
			if (Mass > 0.0f)
			{
				World.Bodies[O.Body].bKinematic = true;
			}
			return Obstacles.Add(O);
		}

		bool FSoftScene::Build(const FRubbleSceneRequest& R, FRubbleSceneAssets& Assets, FString& Error)
		{
			A = &Assets;
			Assets.Stage = ERubbleStage::CrashHall;
			Assets.GroundHalfExtentCm = 6000.0f;
			FBodyBuilder B{ World, Assets };
			const FString& Id = R.Id;
			const FLinearColor Blue(0.04f, 0.16f, 0.5f), Red(0.55f, 0.04f, 0.03f), Silver(0.55f, 0.56f, 0.58f);
			const FLinearColor Concrete(0.62f, 0.6f, 0.57f);
			const float Speed = R.Param(TEXT("speed"), 50.0f) * KmhToCms;
			auto Car = [&](const FVector2D& At, float Yaw, float InSpeed, const FLinearColor& Paint, bool bPlayer = false) {
				// The hubs sit at the wheel radius above the core's origin: at rest the origin is
				// the static sag (6 cm) below the ground.
				const int32 c = AddCar(B, FVector(At, -6.0), Yaw, 0.0f, Paint, bPlayer);
				Softs[c].Drive = InSpeed;
				Softs[c].Vel = Softs[c].Rot.GetAxisX() * InSpeed;
				return c;
			};
			if (Id == TEXT("yard"))
			{
				Assets.Stage = ERubbleStage::Yard;
				Car(FVector2D(0, 0), 0.0f, 0.0f, Blue, true);
				AddObstacle(B, FVector(2500, 0, 100), FVector(60, 600, 100), FQuat::Identity, ERubbleMaterial::Concrete, Concrete);
				AddObstacle(B, FVector(0, 2500, 150), FVector(10, 10, 150), FQuat::Identity, ERubbleMaterial::Steel, Silver);
				AddObstacle(B, FVector(-2000, -1500, 15), FVector(400, 15, 15), FQuat::Identity, ERubbleMaterial::Concrete, Concrete);
				for (int32 i = 0; i < 6; ++i)
				{
					AddCrate(B, FVector(1200 + (i % 3) * 140.0, -800 + (i / 3) * 140.0, 70 + (i / 3) * 140.0), 60.0f);
				}
			}
			else if (Id == TEXT("wall") || Id == TEXT("import"))
			{
				const float Overlap = FMath::Clamp(R.Param(TEXT("overlap"), 1.0f), 0.1f, 1.0f);
				const float Angle = FMath::DegreesToRadians(R.Param(TEXT("angle"), 0.0f));
				Car(FVector2D(-1200, 0), 0.0f, Speed, Blue, true);
				// The wall covers Overlap of the car's width from its left side (-Y).
				const float Width = 180.0f * Overlap;
				AddObstacle(B, FVector(260, -90 + Width * 0.5 - (Overlap < 1.0f ? 0.0f : 400.0f), 150), FVector(40, Overlap < 1.0f ? Width * 0.5 : 600.0f, 150), FQuat(FVector::UpVector, Angle), ERubbleMaterial::Concrete, Concrete);
			}
			else if (Id == TEXT("pole") || Id == TEXT("sidepole"))
			{
				const bool bSide = Id == TEXT("sidepole");
				const float Offset = R.Param(TEXT("offset"), 0.0f) * 100.0f;
				if (bSide)
				{
					const int32 c = Car(FVector2D(0, -900), 0.0f, 0.0f, Blue, true);
					Softs[c].Vel = FVector(0, Speed, 0); // sliding sideways into the pole
					Softs[c].bNoGrip = true;
				}
				else
				{
					Car(FVector2D(-1200, 0), 0.0f, Speed, Blue, true);
				}
				AddObstacle(B, FVector(bSide ? 20 : 300, bSide ? 120 : Offset, 250), FVector(10, 10, 250), FQuat::Identity, ERubbleMaterial::Steel, Silver);
			}
			else if (Id == TEXT("tbone"))
			{
				const float At = R.Param(TEXT("at"), 0.0f) * 100.0f;
				const float Mass = R.Param(TEXT("mass"), 1500.0f);
				Car(FVector2D(0, 0), 0.0f, R.Param(TEXT("carSpeed"), 0.0f) * KmhToCms, Blue, true);
				Ram = AddObstacle(B, FVector(At, -900, 60), FVector(80, 60, 55), FQuat::Identity, ERubbleMaterial::Steel, FLinearColor(0.75f, 0.6f, 0.1f), Mass, FVector(0, Speed, 0));
			}
			else if (Id == TEXT("headon"))
			{
				const float Overlap = FMath::Clamp(R.Param(TEXT("overlap"), 1.0f), 0.1f, 1.0f);
				const float Shift = 180.0f * (1.0f - Overlap);
				Car(FVector2D(-800, -Shift * 0.5f), 0.0f, Speed, Blue, true);
				Car(FVector2D(800, Shift * 0.5f), UE_PI, Speed, Red);
			}
			else if (Id == TEXT("rollover"))
			{
				const int32 c = Car(FVector2D(0, -900), 0.0f, 0.0f, Blue, true);
				Softs[c].Vel = FVector(0, Speed, 0);
				Softs[c].bNoGrip = true;
				AddObstacle(B, FVector(0, 120, 9), FVector(600, 15, 9), FQuat::Identity, ERubbleMaterial::Concrete, Concrete);
			}
			else if (Id == TEXT("ramp"))
			{
				Car(FVector2D(-2500, 0), 0.0f, Speed, Blue, true);
				// Under the left wheels only: a twisting ramp (pitched up, rolled toward the car).
				AddObstacle(B, FVector(-500, -90, 40), FVector(600, 70, 40), FQuat(FVector(1, 0, 0), -0.25) * FQuat(FVector(0, 1, 0), -0.12), ERubbleMaterial::Concrete, Concrete);
			}
			else if (Id == TEXT("drop"))
			{
				const int32 c = AddCar(B, FVector(0, 0, R.Param(TEXT("height"), 8.0f) * 100.0f), 0.0f, FMath::DegreesToRadians(R.Param(TEXT("pitch"), 50.0f)), Blue, true);
				Softs[c].Vel = FVector::ZeroVector;
			}
			else if (Id == TEXT("crates"))
			{
				for (int32 i = 0; i < 12; ++i)
				{
					AddCrate(B, FVector((i % 4 - 1.5) * 220.0, (i / 4 - 1) * 220.0, 300 + i * 90.0), 60.0f);
				}
			}
			else if (Id == TEXT("traffic"))
			{
				Assets.Stage = ERubbleStage::Yard;
				const int32 N = FMath::Clamp(FMath::RoundToInt32(R.Param(TEXT("cars"), 20.0f)), 1, 100);
				for (int32 i = 0; i < N; ++i)
				{
					// Two crossing loops: cars meet at the crossings.
					const int32 Loop = i & 1;
					const FVector2D Center(Loop ? 1200.0 : -1200.0, 0.0);
					const float Radius = 1500.0f + (i % 3) * 250.0f;
					const float Angle = UE_TWO_PI * float(i / 2) / FMath::Max(1, (N + 1) / 2) + (Loop ? 0.0f : 0.4f);
					const FVector2D At = Center + FVector2D(Radius * FMath::Cos(Angle), Radius * FMath::Sin(Angle));
					const int32 c = Car(At, Angle + (Loop ? UE_HALF_PI : -UE_HALF_PI), World.Random.FRandRange(900.0f, 1500.0f), FLinearColor::MakeFromHSV8(uint8(World.Random.RandHelper(255)), 190, 190), i == 0);
					Softs[c].bAI = true;
					Softs[c].Loop = Center;
					Softs[c].LoopRadius = Radius * (Loop ? 1.0f : -1.0f);
				}
				Assets.GroundHalfExtentCm = 9000.0f;
			}
			else if (Id == TEXT("limbo"))
			{
				Car(FVector2D(-1500, 0), 0.0f, Speed, Blue, true);
				const float Bar = R.Param(TEXT("bar"), 1.2f) * 100.0f;
				AddObstacle(B, FVector(300, 0, Bar + 15), FVector(15, 300, 15), FQuat::Identity, ERubbleMaterial::Steel, FLinearColor(0.85f, 0.1f, 0.1f));
				AddObstacle(B, FVector(300, -300, (Bar + 30) * 0.5f), FVector(15, 15, (Bar + 30) * 0.5f), FQuat::Identity, ERubbleMaterial::Steel, Silver);
				AddObstacle(B, FVector(300, 300, (Bar + 30) * 0.5f), FVector(15, 15, (Bar + 30) * 0.5f), FQuat::Identity, ERubbleMaterial::Steel, Silver);
			}
			else if (Id == TEXT("sandwich"))
			{
				Car(FVector2D(0, 0), 0.0f, 0.0f, Blue, true);
				AddObstacle(B, FVector(600, 0, 160), FVector(300, 130, 160), FQuat::Identity, ERubbleMaterial::Steel, FLinearColor(0.2f, 0.3f, 0.6f));
				AddObstacle(B, FVector(-1500, 0, 160), FVector(350, 130, 160), FQuat::Identity, ERubbleMaterial::Steel, FLinearColor(0.7f, 0.2f, 0.1f), 18000.0f, FVector(Speed, 0, 0));
			}
			else if (Id == TEXT("crusher"))
			{
				Car(FVector2D(0, 0), 0.0f, 0.0f, Blue, true);
				const int32 o = AddObstacle(B, FVector(0, 0, 150 + R.Param(TEXT("height"), 4.0f) * 100.0f), FVector(150, 150, 60), FQuat::Identity, ERubbleMaterial::Concrete,
					FLinearColor(0.4f, 0.4f, 0.42f), R.Param(TEXT("mass"), 10000.0f));
				Obstacles[o].bGravity = true;
			}
			else
			{
				Error = FString::Printf(TEXT("no soft scene '%s'"), *Id);
				return false;
			}
			for (int32 i = 0; i < 8; ++i)
			{
				Balls.Add(B.ParkedBall(20.0f));
			}
			// What a real backend needs to spawn the same scene: each body's start pose and
			// velocity, and the moving obstacles' velocity and mass.
			for (const FSoft& S : Softs)
			{
				FRubbleSoftVisual& V = Assets.SoftBodies[S.Visual];
				V.Spawn = FTransform(S.Rot, S.Pos);
				V.Velocity = S.Vel;
				V.bCar = S.bWheels;
				V.bAI = S.bAI;
			}
			for (const FObb& O : Obstacles)
			{
				if (O.Mass > 0.0f && Assets.Bodies.IsValidIndex(O.Body))
				{
					Assets.Bodies[O.Body].InitialVelocity = O.Vel;
					Assets.Bodies[O.Body].Mass = O.Mass;
					Assets.Bodies[O.Body].bAwake = true;
				}
			}
			const FVector Focus = Softs.Num() ? Softs[0].Pos : FVector::ZeroVector;
			Assets.CameraTarget = Focus + FVector(400, 0, 80);
			Assets.CameraEye = Focus + FVector(-200, -900, 350);
			return true;
		}

		void FSoftScene::SideRam()
		{
			// softbody.html's T: a 1,500 kg ram launched at the player's left side at 40 km/h.
			if (!Softs.Num())
			{
				return;
			}
			const FSoft& S = Softs[0];
			const FVector Left = -S.Rot.GetAxisY();
			if (Ram == INDEX_NONE)
			{
				// Assets are fixed after load: a ram added late would not be drawn, so it is parked
				// at load in scenes that use it; elsewhere T reuses a ball.
				Launch(S.Pos + Left * 900.0 + FVector(0, 0, 60), -Left);
				return;
			}
			FObb& O = Obstacles[Ram];
			O.Center = S.Pos + Left * 900.0 + FVector(0, 0, 60);
			O.Vel = -Left * 40.0f * KmhToCms;
		}

		void FSoftScene::Launch(const FVector& Origin, const FVector& Direction)
		{
			if (Balls.Num())
			{
				FBodyBuilder::Fire(World, Balls[NextBall], Origin, Direction, 3000.0f);
				NextBall = (NextBall + 1) % Balls.Num();
			}
		}

		float FSoftScene::GroundAt(const FVector& P) const
		{
			float Z = 0.0f;
			for (const FObb& O : Obstacles)
			{
				float Top;
				if (O.Mass == 0.0f && TopOf(O, P, Top) && Top < P.Z + 40.0f)
				{
					Z = FMath::Max(Z, Top);
				}
			}
			return Z;
		}

		// ----- stepping -------------------------------------------------------------------------

		void FSoftScene::StepSoft(int32 Index, float Dt, FRubbleFrame& Out)
		{
			FSoft& S = Softs[Index];
			if (S.bAsleep)
			{
				return;
			}
			S.BreaksThisStep = 0;
			const float CrushLength = S.bWheels ? 70.0f : 40.0f;

			// Drive: the player's input, AI loops, or a scripted speed held until the first contact.
			bool bGrounded = false;
			const FTransform T0(S.Rot, S.Pos);
			if (S.bWheels)
			{
				int32 Down = 0;
				for (int32 w = 0; w < S.Wheels.Num(); ++w)
				{
					const FRubbleWheelDef& W = S.Wheels[w];
					const FVector Hub = T0.TransformPosition(FVector(W.LocalHub));
					const float G = GroundAt(Hub);
					const float Pen = G + W.Radius - float(Hub.Z);
					if (Pen > -4.0f)
					{
						++Down;
						// Spring and damper at the hub: four wheels hold the car at SagCm, damped
						// a little under critical.
						constexpr float SagCm = 6.0f;
						const FVector R = Hub - S.Pos;
						const FVector VHub = S.Vel + FVector::CrossProduct(S.AngVel, R);
						const float Accel = FMath::Max(0.0f, 0.25f * GravityCm * FMath::Max(Pen, 0.0f) / SagCm - 4.0f * float(VHub.Z));
						const FVector Dv(0, 0, Accel * Dt);
						S.Vel += Dv;
						S.AngVel += FVector::CrossProduct(R, Dv) * (S.Mass / S.Inertia);
					}
				}
				bGrounded = Down >= 3;
			}
			S.Vel.Z -= GravityCm * Dt;

			if (S.bNoGrip && S.Vel.Size2D() < 100.0)
			{
				S.bNoGrip = false;
			}
			if (bGrounded && !S.bNoGrip)
			{
				const FVector Fwd = S.Rot.GetAxisX().GetSafeNormal2D();
				float Want = S.Drive, Steer = 0.0f;
				if (S.bPlayer && !S.bAI && S.Drive == 0.0f)
				{
					Want = Input.Throttle * 2500.0f;
					Steer = Input.Steer;
				}
				else if (S.bAI)
				{
					const FVector2D P(S.Pos);
					const float Dir = FMath::Sign(S.LoopRadius);
					const FVector2D Rel = P - S.Loop;
					const float Angle = FMath::Atan2(float(Rel.Y), float(Rel.X)) + Dir * 0.3f;
					const FVector2D Aim = S.Loop + FVector2D(FMath::Cos(Angle), FMath::Sin(Angle)) * FMath::Abs(S.LoopRadius);
					const float Yaw = FMath::Atan2(float(Fwd.Y), float(Fwd.X));
					Steer = FMath::Clamp(FMath::FindDeltaAngleRadians(Yaw, FMath::Atan2(float(Aim.Y - P.Y), float(Aim.X - P.X))) * 2.0f, -1.0f, 1.0f);
				}
				S.Steer = FMath::FInterpTo(S.Steer, Steer, Dt, 8.0f);
				// Tyres: forward speed toward the wanted one, side slip bled off.
				const FVector Side(-Fwd.Y, Fwd.X, 0.0);
				float Vf = float(FVector::DotProduct(S.Vel, Fwd));
				float Vs = float(FVector::DotProduct(S.Vel, Side));
				const bool bScripted = S.Drive != 0.0f || S.bAI;
				if (bScripted || Input.Throttle != 0.0f || !S.bPlayer)
				{
					Vf = FMath::FInterpTo(Vf, Want, Dt, bScripted ? 4.0f : 1.2f);
				}
				if (S.bPlayer && Input.Brake > 0.0f)
				{
					Vf = FMath::FInterpTo(Vf, 0.0f, Dt, 3.0f);
				}
				Vs = FMath::FInterpTo(Vs, 0.0f, Dt, 6.0f);
				S.Vel = Fwd * Vf + Side * Vs + FVector(0, 0, S.Vel.Z);
				S.AngVel.Z = Vf / 273.0f * FMath::Tan(S.Steer * 0.6f);
				S.AngVel.X *= 0.97;
				S.AngVel.Y *= 0.97;
			}

			// Integrate the core.
			S.Pos += S.Vel * Dt;
			const double W = S.AngVel.Size();
			if (W > 1e-6)
			{
				S.Rot = (FQuat(S.AngVel / W, W * Dt) * S.Rot).GetNormalized();
			}

			// Contacts: every vertex against the ground, the obstacles and the other soft bodies.
			const FTransform T(S.Rot, S.Pos);
			TArray<FObb, TInlineAllocator<16>> Near;
			const float Reach = S.Half.Size() + 50.0f;
			for (const FObb& O : Obstacles)
			{
				if (FVector::Dist(O.Center, S.Pos) < Reach + O.Half.Size())
				{
					Near.Add(O);
				}
			}
			for (int32 o = 0; o < Softs.Num(); ++o)
			{
				const FSoft& Other = Softs[o];
				if (o != Index && FVector::Dist(Other.Pos, S.Pos) < Reach + Other.Half.Size())
				{
					FObb Core;
					Core.Center = Other.Rot.RotateVector(FVector(0, 0, Other.bWheels ? 70.0 : 0.0)) + Other.Pos;
					Core.Rot = Other.Rot;
					Core.Half = FVector(Other.Half) * FVector(0.92, 0.92, Other.bWheels ? 0.6 : 0.9);
					Core.Vel = Other.Vel;
					Core.Mass = Other.Mass;
					Near.Add(Core);
				}
			}
			FVector Deepest = FVector::ZeroVector, DeepestN = FVector::UpVector, DeepestVel = FVector::ZeroVector;
			float DeepestD = 0.0f, DeepestMass = 0.0f;
			bool bContact = false;
			const FQuat InvRot = S.Rot.Inverse();
			for (int32 v = 0; v < S.Rest.Num(); ++v)
			{
				if (S.PieceOf[S.Part[v]] >= 0)
				{
					continue; // detached: moves with its piece
				}
				FVector P = T.TransformPosition(FVector(S.Rest[v] + S.Offset[v]));
				FVector Push = FVector::ZeroVector;
				FVector PushN = FVector::UpVector;
				FVector PushVel = FVector::ZeroVector;
				float PushMass = 0.0f;
				if (P.Z < 0.0)
				{
					Push = FVector(0, 0, -P.Z);
				}
				for (const FObb& O : Near)
				{
					const FVector L = O.Rot.UnrotateVector(P - O.Center);
					const FVector Pen = O.Half - L.GetAbs();
					if (Pen.X > 0 && Pen.Y > 0 && Pen.Z > 0)
					{
						const int32 Ax = Pen.X < Pen.Y ? (Pen.X < Pen.Z ? 0 : 2) : (Pen.Y < Pen.Z ? 1 : 2);
						FVector N = FVector::ZeroVector;
						N[Ax] = FMath::Sign(L[Ax]);
						N = O.Rot.RotateVector(N);
						const FVector Pv = N * Pen[Ax];
						if (Pv.SizeSquared() > Push.SizeSquared())
						{
							Push = Pv;
							PushN = N;
							PushVel = O.Vel;
							PushMass = O.Mass;
						}
					}
				}
				if (Push.IsNearlyZero())
				{
					continue;
				}
				bContact = true;
				const float d = float(Push.Size());
				if (d > DeepestD)
				{
					DeepestD = d;
					Deepest = P;
					DeepestN = Push / d;
					DeepestVel = PushVel;
					DeepestMass = PushMass;
				}
				// Plastic: the vertex stays where it was pushed, wrinkled a little along its normal.
				// A ground contact (the push is straight up from below Z = 0) only dents on a hard landing.
				const bool bGround = P.Z < 1.0 && Push.X == 0.0 && Push.Y == 0.0 && Push.Z > 0.0;
				if (!bGround || d > 2.0f)
				{
					const FVector3f Local = FVector3f(InvRot.RotateVector(Push));
					const FVector3f N = FVector3f(InvRot.RotateVector(Push / d));
					S.Offset[v] += Local + FVector3f::CrossProduct(N, FVector3f(0, 0, 1)).GetSafeNormal() * (S.Noise[v] * 0.25f * d);
					S.bChanged = true;
					S.PartCrush[S.Part[v]] = FMath::Max(S.PartCrush[S.Part[v]], S.Offset[v].Size());
					if (d > 4.0f && S.BreaksThisStep < 3 && World.Random.FRand() < 0.02f)
					{
						++S.BreaksThisStep;
						FRubbleEvent E;
						E.Type = ERubbleEventType::BeamBroke;
						E.Position = P;
						E.Normal = Push / d;
						E.Magnitude = d * 2000.0f;
						E.Soft = S.Visual;
						E.Part = PartName(S.Part[v]);
						Out.Events.Add(E);
					}
				}
			}
			if (bContact && DeepestD > 3.0f && Deepest.Z > 2.0 && !S.bAI)
			{
				// A crash test drives up to the first hard contact, then coasts.
				S.Drive = 0.0f;
			}
			if (bContact)
			{
				// Crush: the core's speed into the contact falls off over CrushLength of crumpling.
				const FVector R = Deepest - S.Pos;
				const FVector Vp = S.Vel + FVector::CrossProduct(S.AngVel, R) - DeepestVel;
				const float Vn = float(FVector::DotProduct(Vp, DeepestN));
				if (Vn < 0.0f)
				{
					const bool bHard = DeepestD > 3.0f;
					const float f = bHard ? FMath::Clamp(DeepestD / CrushLength + 0.04f, 0.0f, 1.0f) : 0.6f;
					const FVector RxN = FVector::CrossProduct(R, DeepestN);
					const float InvMass = 1.0f / S.Mass + (DeepestMass > 0.0f ? 1.0f / DeepestMass : 0.0f);
					const float Jm = -f * Vn / (InvMass + float(RxN.SizeSquared()) / S.Inertia);
					S.Vel += DeepestN * (Jm / S.Mass);
					S.AngVel += RxN * (Jm / S.Inertia);
					// Friction along the contact.
					const FVector Vt = Vp - DeepestN * Vn;
					S.Vel -= Vt * FMath::Min(1.0f, 0.4f * f);
					if (-Vn > 300.0f && bHard)
					{
						FRubbleEvent E;
						E.Type = ERubbleEventType::Impact;
						E.Position = Deepest;
						E.Normal = DeepestN;
						E.Magnitude = -Vn;
						E.MaterialA = ERubbleMaterial::CarPaint;
						E.MaterialB = ERubbleMaterial::Concrete;
						Out.Events.Add(E);
					}
					// The other side takes the rest (rams and cars carry on slower).
					for (FObb& O : Obstacles)
					{
						if (O.Mass > 0.0f && FVector::Dist(O.Center, Deepest) < O.Half.Size() + 60.0f)
						{
							O.Vel -= DeepestN * (Jm / O.Mass);
						}
					}
				}
			}

			// Crumple zone: plastic flow spreads to neighbours, falling off with each edge.
			if (S.bChanged)
			{
				for (int32 v = 0; v < S.Rest.Num(); ++v)
				{
					const float Mine = S.Offset[v].SizeSquared();
					if (Mine < 1.0f)
					{
						continue;
					}
					for (const int32 n : S.Adj[v])
					{
						if (S.PieceOf[S.Part[n]] < 0 && S.PieceOf[S.Part[v]] < 0)
						{
							// About 0.9 m of crumple zone around a hit (0.72 per ~8 cm edge).
							const FVector3f Spread = S.Offset[v] * 0.72f;
							if (Spread.SizeSquared() > S.Offset[n].SizeSquared() * 1.05f)
							{
								S.Offset[n] = FMath::Lerp(S.Offset[n], Spread, 0.35f);
							}
						}
					}
				}
			}

			// Parts tear off once crushed or pulled enough: a piece flies with the core's velocity.
			for (uint8 p = 1; p < PartCount; ++p)
			{
				if (S.PieceOf[p] < 0 && S.PartCrush[p] > (p == PartRoof ? 22.0f : 30.0f))
				{
					FPiece Piece;
					Piece.Pos = S.Pos;
					Piece.Rot = S.Rot;
					Piece.Vel = S.Vel * 0.8 + FVector(0, 0, 200) + World.Random.VRand() * 150.0;
					Piece.AngVel = World.Random.VRand() * 3.0;
					S.PieceOf[p] = int8(S.Pieces.Add(Piece));
					FRubbleEvent E;
					E.Type = ERubbleEventType::PieceDetached;
					E.Soft = S.Visual;
					E.Part = PartName(p);
					E.Position = S.Pos;
					E.MaterialA = ERubbleMaterial::Glass;
					Out.Events.Add(E);
				}
			}
			for (FPiece& Piece : S.Pieces)
			{
				if (Piece.bResting)
				{
					continue;
				}
				Piece.Vel.Z -= GravityCm * Dt;
				Piece.Pos += Piece.Vel * Dt;
				const double PW = Piece.AngVel.Size();
				if (PW > 1e-6)
				{
					Piece.Rot = (FQuat(Piece.AngVel / PW, PW * Dt) * Piece.Rot).GetNormalized();
				}
				S.bChanged = true;
			}
			// Pieces land: their lowest vertex on the ground.
			if (S.Pieces.Num())
			{
				TArray<float> Lowest;
				Lowest.Init(1e9f, S.Pieces.Num());
				for (int32 v = 0; v < S.Rest.Num(); ++v)
				{
					const int8 pi = S.PieceOf[S.Part[v]];
					if (pi >= 0)
					{
						const FPiece& Piece = S.Pieces[pi];
						Lowest[pi] = FMath::Min(Lowest[pi], float(FTransform(Piece.Rot, Piece.Pos).TransformPosition(FVector(S.Rest[v] + S.Offset[v])).Z));
					}
				}
				for (int32 pi = 0; pi < S.Pieces.Num(); ++pi)
				{
					FPiece& Piece = S.Pieces[pi];
					if (Lowest[pi] < 0.0f && !Piece.bResting)
					{
						Piece.Pos.Z -= Lowest[pi];
						Piece.Vel *= FVector(0.6, 0.6, -0.2);
						Piece.AngVel *= 0.6;
						if (Piece.Vel.Size() < 40.0)
						{
							Piece.bResting = true;
						}
					}
				}
			}

			// Moving obstacles that this body drives (rams, the shunting truck, the crusher).
			const bool bMoving = S.Vel.SizeSquared() > 15.0 * 15.0 || S.AngVel.SizeSquared() > 0.05 * 0.05;
			S.Still = (bMoving || (bContact && DeepestD > 0.5f)) ? 0.0f : S.Still + Dt;
			if (S.Still > 1.0f && !S.bPlayer)
			{
				S.bAsleep = true;
			}

			// Skin output: positions, normals, strain, torn triangles.
			if (S.bChanged || bMoving)
			{
				FRubbleSoftSkin Skin;
				Skin.Soft = S.Visual;
				Skin.Positions.SetNumUninitialized(S.Rest.Num());
				Skin.Strain.SetNumUninitialized(S.Rest.Num());
				for (int32 v = 0; v < S.Rest.Num(); ++v)
				{
					const int8 pi = S.PieceOf[S.Part[v]];
					const FTransform Frame = pi >= 0 ? FTransform(S.Pieces[pi].Rot, S.Pieces[pi].Pos) : T;
					Skin.Positions[v] = FVector3f(Frame.TransformPosition(FVector(S.Rest[v] + S.Offset[v])));
					Skin.Strain[v] = FMath::Clamp(S.Offset[v].Size() / 25.0f, 0.0f, 1.0f);
				}
				ComputeNormals(Skin.Positions, S.Indices, Skin.Normals);
				Skin.TornTriangles.Init(false, S.Indices.Num() / 3);
				for (int32 t = 0; t < S.Indices.Num(); t += 3)
				{
					for (int32 k = 0; k < 3; ++k)
					{
						const float L = FVector3f::Dist(Skin.Positions[S.Indices[t + k]], Skin.Positions[S.Indices[t + (k + 1) % 3]]);
						if (L > 1.8f * S.RestEdge[t + k] + 2.0f)
						{
							Skin.TornTriangles[t / 3] = true;
						}
					}
				}
				Out.Skins.Add(MoveTemp(Skin));
				S.bChanged = false;
			}

			// Wheels: hub poses on the core (they stay with it; the bodywork crumples around them).
			for (int32 w = 0; w < S.Wheels.Num(); ++w)
			{
				const FRubbleWheelDef& Def = S.Wheels[w];
				const FVector Hub = T.TransformPosition(FVector(Def.LocalHub));
				const float G = GroundAt(Hub);
				const float Pen = FMath::Clamp(G + Def.Radius - float(Hub.Z), 0.0f, 12.0f);
				const bool bFront = Def.LocalHub.X > 0.0f;
				const float Vf = float(FVector::DotProduct(S.Vel, S.Rot.GetAxisX()));
				S.Spin[w] = FMath::Fmod(S.Spin[w] + Vf / Def.Radius * Dt, UE_TWO_PI);
				FRubbleWheelVisual WV;
				WV.Vehicle = S.Visual;
				WV.bSoft = true;
				WV.Wheel = w;
				WV.SteerRad = bFront ? S.Steer * 0.6f : 0.0f;
				WV.Hub = FTransform(S.Rot * FQuat(FVector::UpVector, WV.SteerRad), Hub + FVector(0, 0, Pen));
				WV.SpinRad = S.Spin[w];
				WV.Compression = Pen / 12.0f;
				WV.bOnGround = Pen > 0.0f;
				Out.Wheels.Add(WV);
			}
		}

		void FSoftScene::Step(float Dt, FRubbleFrame& Out)
		{
			Time += Dt;
			for (FObb& O : Obstacles)
			{
				if (O.Mass <= 0.0f)
				{
					continue;
				}
				if (O.bGravity)
				{
					O.Vel.Z -= GravityCm * Dt;
				}
				O.Center += O.Vel * Dt;
				if (O.Center.Z - O.Half.Z < 0.0)
				{
					O.Center.Z = O.Half.Z;
					O.Vel = FVector::ZeroVector;
				}
				O.Vel *= 0.9995;
				if (O.Body != INDEX_NONE)
				{
					FBody& Body = World.Bodies[O.Body];
					if (!Body.bReportedOnce || !Body.Com.Equals(O.Center, 0.01))
					{
						Body.Com = O.Center;
						Body.Rot = O.Rot;
						World.Report(O.Body, Out);
					}
				}
			}
			for (int32 i = 0; i < Softs.Num(); ++i)
			{
				StepSoft(i, Dt, Out);
			}
			World.Step(Dt, Out);
		}
	}

	TUniquePtr<FScene> MakeSoftScene()
	{
		return TUniquePtr<FScene>(new FSoftScene());
	}
}

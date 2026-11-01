#include "Mock/MockShared.h"

namespace RubbleMock
{
	int32 FWorld::Add(const FBody& Body)
	{
		const int32 Index = Bodies.Add(Body);
		if (Body.bAwake && !Body.bStatic && !Body.bKinematic)
		{
			Awake.Add(Index);
		}
		return Index;
	}

	namespace
	{
		constexpr double HeightCell = 40.0;
		FIntPoint CellOf(double X, double Y) { return FIntPoint(FMath::FloorToInt32(X / HeightCell), FMath::FloorToInt32(Y / HeightCell)); }
	}

	double FWorld::GroundAt(const FVector& P, double Radius) const
	{
		if (Heights.Num() == 0)
		{
			return 0.0;
		}
		const FIntPoint Lo = CellOf(P.X - Radius, P.Y - Radius), Hi = CellOf(P.X + Radius, P.Y + Radius);
		float Z = 0.0f;
		for (int32 x = Lo.X; x <= Hi.X; ++x)
		{
			for (int32 y = Lo.Y; y <= Hi.Y; ++y)
			{
				if (const float* H = Heights.Find(FIntPoint(x, y)))
				{
					// Only what is below the body's centre holds it up.
					Z = *H < P.Z ? FMath::Max(Z, *H) : Z;
				}
			}
		}
		return Z;
	}

	void FWorld::Settle(int32 Index)
	{
		const FBody& B = Bodies[Index];
		if (B.bProjectile)
		{
			return;
		}
		const FVector AX = B.Rot.GetAxisX(), AY = B.Rot.GetAxisY(), AZ = B.Rot.GetAxisZ();
		const double Up = FMath::Abs(AX.Z) * B.HalfExtents.X + FMath::Abs(AY.Z) * B.HalfExtents.Y + FMath::Abs(AZ.Z) * B.HalfExtents.Z;
		const double Ex = FMath::Abs(AX.X) * B.HalfExtents.X + FMath::Abs(AY.X) * B.HalfExtents.Y + FMath::Abs(AZ.X) * B.HalfExtents.Z;
		const double Ey = FMath::Abs(AX.Y) * B.HalfExtents.X + FMath::Abs(AY.Y) * B.HalfExtents.Y + FMath::Abs(AZ.Y) * B.HalfExtents.Z;
		FRaised R;
		R.Top = float(B.Com.Z + Up);
		// The footprint, a little inside the box (so neighbours do not perch on its edge).
		const FIntPoint Lo = CellOf(B.Com.X - 0.7 * Ex, B.Com.Y - 0.7 * Ey), Hi = CellOf(B.Com.X + 0.7 * Ex, B.Com.Y + 0.7 * Ey);
		for (int32 x = Lo.X; x <= Hi.X; ++x)
		{
			for (int32 y = Lo.Y; y <= Hi.Y; ++y)
			{
				float& H = Heights.FindOrAdd(FIntPoint(x, y));
				R.Cells.Add({ FIntPoint(x, y), H });
				H = FMath::Max(H, R.Top);
			}
		}
		RaisedBy.Add(Index, MoveTemp(R));
	}

	void FWorld::Unsettle(int32 Index)
	{
		FRaised R;
		if (!RaisedBy.RemoveAndCopyValue(Index, R))
		{
			return;
		}
		for (const TPair<FIntPoint, float>& C : R.Cells)
		{
			float* H = Heights.Find(C.Key);
			if (H && *H <= R.Top + 0.01f)
			{
				*H = C.Value;
			}
		}
	}

	void FWorld::Wake(int32 Index, const FVector& DeltaVel)
	{
		FBody& B = Bodies[Index];
		if (B.bStatic || B.bKinematic || !B.bAlive)
		{
			return;
		}
		Unsettle(Index);
		B.Vel += DeltaVel;
		B.RestTime = 0.0f;
		if (!B.bAwake)
		{
			B.bAwake = true;
			B.bWokeThisStep = true;
			Awake.Add(Index);
			Woken.Add(Index);
		}
	}

	void FWorld::Blast(const FVector& Point, float Radius, float Speed, const FVector& Along)
	{
		const double R2 = double(Radius) * Radius;
		for (int32 i = 0; i < Bodies.Num(); ++i)
		{
			const FBody& B = Bodies[i];
			if (B.bStatic || B.bKinematic || B.bProjectile || !B.bAlive)
			{
				continue;
			}
			const FVector D = B.Com - Point;
			const double D2 = D.SizeSquared();
			if (D2 > R2)
			{
				continue;
			}
			const double Falloff = 1.0 - FMath::Sqrt(D2 / R2);
			const FVector Dir = (D.GetSafeNormal() * 0.6 + Along * 0.6 + FVector(0, 0, 0.25)).GetSafeNormal();
			Wake(i, Dir * Speed * Falloff * Random.FRandRange(0.6f, 1.2f));
			Bodies[i].AngVel += Random.VRand() * 6.0 * Falloff;
		}
	}

	void FWorld::BuildSupports()
	{
		// Hash bodies by their (x, y) cell; a body supports another whose bottom sits on its top.
		const double Cell = 120.0;
		TMultiMap<FIntPoint, int32> Grid;
		auto Key = [Cell](const FVector& P) { return FIntPoint(FMath::FloorToInt32(P.X / Cell), FMath::FloorToInt32(P.Y / Cell)); };
		for (int32 i = 0; i < Bodies.Num(); ++i)
		{
			if (!Bodies[i].bStatic && !Bodies[i].bProjectile)
			{
				Grid.Add(Key(Bodies[i].Com), i);
			}
		}
		TArray<int32> Near;
		for (int32 i = 0; i < Bodies.Num(); ++i)
		{
			FBody& Upper = Bodies[i];
			if (Upper.bStatic || Upper.bProjectile)
			{
				continue;
			}
			const FIntPoint K = Key(Upper.Com);
			const double Bottom = Upper.Com.Z - Upper.HalfExtents.Z;
			for (int32 dx = -1; dx <= 1; ++dx)
			{
				for (int32 dy = -1; dy <= 1; ++dy)
				{
					Near.Reset();
					Grid.MultiFind(FIntPoint(K.X + dx, K.Y + dy), Near);
					for (const int32 j : Near)
					{
						if (j == i)
						{
							continue;
						}
						const FBody& Lower = Bodies[j];
						const double Top = Lower.Com.Z + Lower.HalfExtents.Z;
						const bool bTouching = FMath::Abs(Top - Bottom) < 0.25 * Upper.HalfExtents.Z + 2.0;
						const bool bOver = FMath::Abs(Lower.Com.X - Upper.Com.X) < Lower.HalfExtents.X + Upper.HalfExtents.X
							&& FMath::Abs(Lower.Com.Y - Upper.Com.Y) < Lower.HalfExtents.Y + Upper.HalfExtents.Y;
						if (bTouching && bOver)
						{
							Bodies[j].Supports.Add(i);
						}
					}
				}
			}
		}
	}

	int32 FWorld::Launch(const FVector& Origin, const FVector& Direction, float Speed, float Radius)
	{
		FBody B;
		B.Com = Origin;
		B.Vel = Direction.GetSafeNormal() * Speed;
		B.Radius = Radius;
		B.HalfExtents = FVector(Radius);
		B.Mass = 50.0f;
		B.bAwake = true;
		B.bProjectile = true;
		B.Material = ERubbleMaterial::Steel;
		B.bWokeThisStep = true;
		return Add(B);
	}

	void FWorld::Report(int32 Index, FRubbleFrame& Out, bool bFellAsleep)
	{
		FBody& B = Bodies[Index];
		FRubbleMove M;
		M.Body = IdOf(Index);
		M.Transform = B.Origin();
		M.Previous = B.bReportedOnce ? B.Reported : M.Transform;
		M.bHasPrevious = B.bReportedOnce;
		M.bFellAsleep = bFellAsleep;
		M.bWoke = B.bWokeThisStep;
		M.UserData = B.UserData;
		B.Reported = M.Transform;
		B.bReportedOnce = true;
		B.bWokeThisStep = false;
		Out.Moves.Add(M);
	}

	void FWorld::Step(float Dt, FRubbleFrame& Out)
	{
		// Supports of bodies woken last step fall now.
		TArray<int32> Wake2 = MoveTemp(Woken);
		Woken.Reset();
		for (const int32 i : Wake2)
		{
			for (const int32 s : Bodies[i].Supports)
			{
				Wake(s, FVector(Random.FRandRange(-20.f, 20.f), Random.FRandRange(-20.f, 20.f), 0.0));
			}
		}

		int32 Write = 0;
		// Awake grows while this runs (a body woken in the step is integrated in it too); Write
		// never passes r, so compacting in place is safe.
		for (int32 r = 0; r < Awake.Num(); ++r)
		{
			const int32 i = Awake[r];
			FBody& B = Bodies[i];
			if (!B.bAlive || !B.bAwake)
			{
				continue;
			}
			B.Age += Dt;
			B.Vel.Z -= GravityCm * Dt;
			B.Com += B.Vel * Dt;
			const double W = B.AngVel.Size();
			if (W > 1e-6)
			{
				B.Rot = (FQuat(B.AngVel / W, W * Dt) * B.Rot).GetNormalized();
			}

			// Ground: the lowest point of the oriented half extents.
			const FVector AX = B.Rot.GetAxisX(), AY = B.Rot.GetAxisY(), AZ = B.Rot.GetAxisZ();
			const double Support = FMath::Abs(AX.Z) * B.HalfExtents.X + FMath::Abs(AY.Z) * B.HalfExtents.Y + FMath::Abs(AZ.Z) * B.HalfExtents.Z;
			bool bGrounded = false;
			const double Floor = GroundAt(B.Com, 0.5 * B.Radius);
			if (B.Com.Z - Support < Floor)
			{
				B.Com.Z = Floor + Support;
				if (B.Vel.Z < -ImpactSpeed)
				{
					FRubbleEvent E;
					E.Type = ERubbleEventType::Impact;
					E.Position = FVector(B.Com.X, B.Com.Y, Floor);
					E.Normal = FVector::UpVector;
					E.Magnitude = float(-B.Vel.Z);
					E.MaterialA = B.Material;
					E.MaterialB = ERubbleMaterial::Concrete;
					Out.Events.Add(E);
				}
				if (B.Vel.Z < 0.0)
				{
					B.Vel.Z = -B.Vel.Z * Restitution;
				}
				B.Vel.X *= 0.86;
				B.Vel.Y *= 0.86;
				B.AngVel *= 0.8;
				bGrounded = true;
				// Tip toward the nearest face-down orientation as it settles.
				if (B.Vel.SizeSquared() < 200.0 * 200.0)
				{
					const FVector Axes[3] = { AX, AY, AZ };
					int32 Best = 0;
					for (int32 a = 1; a < 3; ++a)
					{
						Best = FMath::Abs(Axes[a].Z) > FMath::Abs(Axes[Best].Z) ? a : Best;
					}
					const FVector Up = Axes[Best] * FMath::Sign(Axes[Best].Z);
					const FQuat Settle = FQuat::FindBetweenNormals(Up, FVector::UpVector);
					B.Rot = FQuat::Slerp(FQuat::Identity, Settle, FMath::Min(1.0, 6.0 * Dt)) * B.Rot;
				}
			}

			if (B.bProjectile)
			{
				Projectiles.Add(i);
			}

			bool bSlept = false;
			const bool bSlow = B.Vel.SizeSquared() < 25.0 * 25.0 && B.AngVel.SizeSquared() < 0.3 * 0.3;
			B.RestTime = bGrounded && bSlow ? B.RestTime + Dt : 0.0f;
			if (B.RestTime > 0.4f)
			{
				B.bAwake = false;
				B.Vel = FVector::ZeroVector;
				B.AngVel = FVector::ZeroVector;
				bSlept = true;
				Settle(i);
			}
			Report(i, Out, bSlept);
			if (!bSlept)
			{
				Awake[Write++] = i;
			}
		}
		Awake.SetNum(Write, EAllowShrinking::No);

		// Projectiles against every body after the integration: a hit may add bodies (a split).
		// A linear scan, fine for a handful of balls.
		for (const int32 i : Projectiles)
		{
			for (int32 j = 0; OnProjectileHit && j < Bodies.Num(); ++j)
			{
				const FBody& P = Bodies[i];
				const FBody& O = Bodies[j];
				if (j == i || O.bProjectile || !O.bAlive || O.bStatic)
				{
					continue;
				}
				const double Reach = O.Radius + P.Radius;
				if (FVector::DistSquared(O.Com, P.Com) < Reach * Reach)
				{
					const FVector Point = O.Com + (P.Com - O.Com).GetSafeNormal() * O.Radius;
					if (OnProjectileHit(i, j, Point))
					{
						Bodies[i].Vel *= 0.45;
						break;
					}
				}
			}
		}
		Projectiles.Reset();
	}

	// ----- geometry ------------------------------------------------------------------------

	void AppendBox(FRubbleMeshData& Mesh, const FVector3f& H, const FVector3f& C, ERubbleMaterial Material)
	{
		static const FVector3f Normals[6] = { { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 } };
		for (const FVector3f& N : Normals)
		{
			// Two tangents with T x B = N; Unreal's front faces are clockwise seen from outside.
			const FVector3f T = FMath::Abs(N.Z) > 0.5f ? FVector3f(1, 0, 0) : FVector3f(0, 0, 1);
			const FVector3f Bt = FVector3f::CrossProduct(N, T);
			const uint32 Base = Mesh.Positions.Num();
			const FVector2f Size(FMath::Abs(FVector3f::DotProduct(T, H)) * 2.0f, FMath::Abs(FVector3f::DotProduct(Bt, H)) * 2.0f);
			for (int32 k = 0; k < 4; ++k)
			{
				const float s = (k == 1 || k == 2) ? 1.0f : -1.0f;
				const float t = (k >= 2) ? 1.0f : -1.0f;
				Mesh.Positions.Add(C + N * H + T * (s * FMath::Abs(FVector3f::DotProduct(T, H))) + Bt * (t * FMath::Abs(FVector3f::DotProduct(Bt, H))));
				Mesh.Normals.Add(N);
				Mesh.UVs.Add(FVector2f((s * 0.5f + 0.5f) * Size.X, (t * 0.5f + 0.5f) * Size.Y) / 100.0f);
			}
			Mesh.Indices.Append({ Base, Base + 2, Base + 1, Base, Base + 3, Base + 2 });
			if (Mesh.TriangleMaterials.Num() || Material != Mesh.Material)
			{
				// Per-triangle materials from here on: the earlier triangles keep the mesh's own.
				const int32 Before = Mesh.NumTriangles() - 2;
				while (Mesh.TriangleMaterials.Num() < Before)
				{
					Mesh.TriangleMaterials.Add(Mesh.Material);
				}
				Mesh.TriangleMaterials.Add(Material);
				Mesh.TriangleMaterials.Add(Material);
			}
		}
	}

	FPolytope FPolytope::Box(const FVector3d& Min, const FVector3d& Max)
	{
		FPolytope P;
		for (int32 i = 0; i < 8; ++i)
		{
			P.Vertices.Add(FVector3d(i & 1 ? Max.X : Min.X, i & 2 ? Max.Y : Min.Y, i & 4 ? Max.Z : Min.Z));
		}
		// Counter-clockwise seen from outside (right-handed sense around the outward normal).
		P.Faces = { { 0, 4, 6, 2 }, { 1, 3, 7, 5 }, { 0, 1, 5, 4 }, { 2, 6, 7, 3 }, { 0, 2, 3, 1 }, { 4, 5, 7, 6 } };
		P.FaceInterior.Init(false, 6);
		return P;
	}

	void FPolytope::Clip(const FVector3d& N, double D)
	{
		constexpr double Eps = 1e-7;
		TArray<double> Dist;
		Dist.SetNum(Vertices.Num());
		bool bAnyOut = false, bAnyIn = false;
		for (int32 i = 0; i < Vertices.Num(); ++i)
		{
			Dist[i] = FVector3d::DotProduct(N, Vertices[i]) - D;
			bAnyOut |= Dist[i] > Eps;
			bAnyIn |= Dist[i] <= Eps;
		}
		if (!bAnyOut)
		{
			return;
		}
		if (!bAnyIn)
		{
			Vertices.Reset();
			Faces.Reset();
			FaceInterior.Reset();
			return;
		}
		TArray<FVector3d> NewVerts;
		TMap<int32, int32> Kept;
		TMap<uint64, int32> EdgeCut;
		auto KeepVertex = [&](int32 v) {
			if (const int32* At = Kept.Find(v))
			{
				return *At;
			}
			return Kept.Add(v, NewVerts.Add(Vertices[v]));
		};
		auto Cut = [&](int32 a, int32 b) {
			const uint64 K = a < b ? (uint64(a) << 32 | uint32(b)) : (uint64(b) << 32 | uint32(a));
			if (const int32* At = EdgeCut.Find(K))
			{
				return *At;
			}
			const double t = Dist[a] / (Dist[a] - Dist[b]);
			return EdgeCut.Add(K, NewVerts.Add(Vertices[a] + (Vertices[b] - Vertices[a]) * t));
		};
		TArray<TArray<int32>> NewFaces;
		TArray<bool> NewInterior;
		// The cap's edges: for each clipped face, the segment between its two cut points.
		TMap<int32, int32> CapNext;
		for (int32 f = 0; f < Faces.Num(); ++f)
		{
			const TArray<int32>& F = Faces[f];
			TArray<int32> Out;
			int32 Enter = INDEX_NONE, Leave = INDEX_NONE;
			for (int32 k = 0; k < F.Num(); ++k)
			{
				const int32 a = F[k], b = F[(k + 1) % F.Num()];
				const bool aIn = Dist[a] <= Eps, bIn = Dist[b] <= Eps;
				if (aIn)
				{
					Out.Add(KeepVertex(a));
				}
				if (aIn != bIn)
				{
					const int32 c = Cut(a, b);
					Out.Add(c);
					(aIn ? Leave : Enter) = c;
				}
			}
			if (Out.Num() >= 3)
			{
				NewFaces.Add(Out);
				NewInterior.Add(FaceInterior[f]);
			}
			if (Leave != INDEX_NONE && Enter != INDEX_NONE)
			{
				// Face winding goes Leave -> (outside) -> Enter; the cap runs the other way.
				CapNext.Add(Enter, Leave);
			}
		}
		if (CapNext.Num() >= 3)
		{
			TArray<int32> Cap;
			int32 Start = CapNext.CreateConstIterator()->Key;
			int32 V = Start;
			for (int32 Guard = 0; Guard <= CapNext.Num(); ++Guard)
			{
				Cap.Add(V);
				const int32* Next = CapNext.Find(V);
				if (!Next || *Next == Start)
				{
					break;
				}
				V = *Next;
			}
			if (Cap.Num() >= 3)
			{
				NewFaces.Add(Cap);
				NewInterior.Add(true);
			}
		}
		Vertices = MoveTemp(NewVerts);
		Faces = MoveTemp(NewFaces);
		FaceInterior = MoveTemp(NewInterior);
	}

	FVector3d FPolytope::Centroid() const
	{
		FVector3d C = FVector3d::ZeroVector;
		for (const FVector3d& V : Vertices)
		{
			C += V;
		}
		return Vertices.Num() ? C / Vertices.Num() : C;
	}

	void PolytopeMesh(const FPolytope& P, const FVector3d& Origin, ERubbleMaterial Exterior, FRubbleMeshData& Out)
	{
		Out.Material = Exterior;
		for (int32 f = 0; f < P.Faces.Num(); ++f)
		{
			const TArray<int32>& F = P.Faces[f];
			const FVector3d A = P.Vertices[F[0]], B = P.Vertices[F[1]], C = P.Vertices[F[2]];
			FVector3d N = FVector3d::CrossProduct(B - A, C - A);
			for (int32 k = 3; k < F.Num() && N.SizeSquared() < 1e-12; ++k)
			{
				N = FVector3d::CrossProduct(B - A, P.Vertices[F[k]] - A);
			}
			N = N.GetSafeNormal();
			const FVector3f N3 = FVector3f(N);
			const FVector3d T = (FMath::Abs(N.Z) > 0.7 ? FVector3d(1, 0, 0) : FVector3d(0, 0, 1)).Cross(N).GetSafeNormal();
			const FVector3d Bt = N.Cross(T);
			const uint32 Base = Out.Positions.Num();
			for (const int32 v : F)
			{
				const FVector3d L = P.Vertices[v] - Origin;
				Out.Positions.Add(FVector3f(L));
				Out.Normals.Add(N3);
				Out.UVs.Add(FVector2f(float(FVector3d::DotProduct(P.Vertices[v], T)), float(FVector3d::DotProduct(P.Vertices[v], Bt))) / 100.0f);
			}
			const ERubbleMaterial M = P.FaceInterior[f] ? ERubbleMaterial::Interior : Exterior;
			for (int32 k = 1; k + 1 < F.Num(); ++k)
			{
				// Counter-clockwise from outside (right-handed) is clockwise in Unreal's left-handed
				// view... Unreal's front face is clockwise as seen with its own axes, which for a
				// right-handed-constructed CCW polygon means reversing: (0, k+1, k).
				Out.Indices.Append({ Base, Base + uint32(k + 1), Base + uint32(k) });
				Out.TriangleMaterials.Add(M);
			}
		}
	}

	void AppendRock(FRubbleMeshData& Mesh, float Radius, FRandomStream& Random)
	{
		const double Phi = (1.0 + FMath::Sqrt(5.0)) / 2.0;
		TArray<FVector3d> V = { { -1, Phi, 0 }, { 1, Phi, 0 }, { -1, -Phi, 0 }, { 1, -Phi, 0 }, { 0, -1, Phi }, { 0, 1, Phi }, { 0, -1, -Phi }, { 0, 1, -Phi }, { Phi, 0, -1 }, { Phi, 0, 1 }, { -Phi, 0, -1 }, { -Phi, 0, 1 } };
		FPolytope P = FPolytope::Box(FVector3d(-Radius), FVector3d(Radius));
		for (const FVector3d& Dir : V)
		{
			const FVector3d N = (Dir.GetSafeNormal() + FVector3d(Random.VRand()) * 0.2).GetSafeNormal();
			P.Clip(N, Radius * Random.FRandRange(0.75f, 0.95f));
		}
		for (bool& b : P.FaceInterior)
		{
			b = false;
		}
		PolytopeMesh(P, FVector3d::ZeroVector, Mesh.Material, Mesh);
		Mesh.TriangleMaterials.Reset();
	}

	void ComputeNormals(const TArray<FVector3f>& Positions, const TArray<uint32>& Indices, TArray<FVector3f>& Normals)
	{
		Normals.Init(FVector3f::ZeroVector, Positions.Num());
		for (int32 t = 0; t + 2 < Indices.Num(); t += 3)
		{
			const uint32 a = Indices[t], b = Indices[t + 1], c = Indices[t + 2];
			// Unreal winding (clockwise front): the outward normal is (c - a) x (b - a).
			const FVector3f N = FVector3f::CrossProduct(Positions[c] - Positions[a], Positions[b] - Positions[a]);
			Normals[a] += N;
			Normals[b] += N;
			Normals[c] += N;
		}
		for (FVector3f& N : Normals)
		{
			N = N.GetSafeNormal(1e-8f, FVector3f::UpVector);
		}
	}
}

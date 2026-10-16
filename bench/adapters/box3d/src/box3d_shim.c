// Thin C API over Box3D for the benchmark adapter (src/main.rs drives it over FFI). Layouts
// below must match the #[repr(C)] mirrors in src/main.rs. One world per process.
#include "box3d/box3d.h"
#include "box3d/collision.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct
{
	uint32_t shape; // 0 box, 1 sphere, 2 hulls, 3 plane
	uint32_t isStatic;
	float half[3]; // box half extents / sphere radius in [0]
	float mass;
	float inertia[3]; // principal, body frame (= centre of mass frame)
	float friction;
	float pos[3];
	float quat[4]; // x y z w
	float linvel[3];
	float angvel[3];
	uint32_t hullFirst, hullCount;
} BxBody;

typedef struct
{
	uint32_t a, b, collide, pad;
	float anchorA[3], anchorB[3];
} BxJoint;

typedef struct
{
	float dt;
	float gravity[3];
	int32_t substeps;
	int32_t workers;
	int32_t sleep;
	int32_t continuous;
	int32_t recycling;
	float contactHertz;      // <0: Box3D default
	float contactDamping;    // <0: default
	float contactSpeed;      // <0: default
	float maxLinearSpeed;
	float planeHalfExtent;   // the ground plane becomes a static box this wide, 1 m thick
} BxSettings;

typedef struct
{
	int32_t hullsIn, hullsChanged, verticesIn, verticesOut, failed;
	char error[256];
} BxReport;

typedef struct
{
	b3WorldId world;
	b3BodyId* bodies;
	int bodyCount;
	b3HullData** hulls; // kept alive for the world's lifetime
	int hullCount;
	b3BoxHull* boxes;
	float* offset; // per body: readback correction (the plane's box sits 1 m below the plane)
	uint8_t* dynamic;
	float dt;
	int substeps;
} BxWorld;

static b3Quat q4( const float* q )
{
	b3Quat r = { { q[0], q[1], q[2] }, q[3] };
	return r;
}

BxWorld* bx_create( const BxSettings* s, const BxBody* bodies, int nb, const uint32_t* hullTable, int nh, const float* verts,
					const BxJoint* joints, int nj, BxReport* report )
{
	memset( report, 0, sizeof( *report ) );
	b3WorldDef def = b3DefaultWorldDef();
	def.gravity = ( b3Vec3 ){ s->gravity[0], s->gravity[1], s->gravity[2] };
	def.enableSleep = s->sleep != 0;
	def.enableContinuous = s->continuous != 0;
	def.workerCount = (uint32_t)( s->workers < 1 ? 1 : s->workers );
	if ( s->contactHertz >= 0.0f ) def.contactHertz = s->contactHertz;
	if ( s->contactDamping >= 0.0f ) def.contactDampingRatio = s->contactDamping;
	if ( s->contactSpeed >= 0.0f ) def.contactSpeed = s->contactSpeed;
	def.maximumLinearSpeed = s->maxLinearSpeed;

	BxWorld* w = (BxWorld*)calloc( 1, sizeof( BxWorld ) );
	w->world = b3CreateWorld( &def );
	w->bodies = (b3BodyId*)calloc( (size_t)nb, sizeof( b3BodyId ) );
	w->hulls = (b3HullData**)calloc( (size_t)( nh > 0 ? nh : 1 ), sizeof( b3HullData* ) );
	w->boxes = (b3BoxHull*)calloc( (size_t)nb, sizeof( b3BoxHull ) );
	w->offset = (float*)calloc( (size_t)nb * 3, sizeof( float ) );
	w->dynamic = (uint8_t*)calloc( (size_t)nb, 1 );
	w->bodyCount = nb;
	w->dt = s->dt;
	w->substeps = s->substeps;

	for ( int i = 0; i < nb; ++i )
	{
		const BxBody* b = bodies + i;
		b3BodyDef bd = b3DefaultBodyDef();
		bd.type = b->isStatic ? b3_staticBody : b3_dynamicBody;
		bd.position = ( b3Pos ){ b->pos[0], b->pos[1], b->pos[2] };
		bd.rotation = q4( b->quat );
		if ( !b->isStatic )
		{
			bd.linearVelocity = ( b3Vec3 ){ b->linvel[0], b->linvel[1], b->linvel[2] };
			bd.angularVelocity = ( b3Vec3 ){ b->angvel[0], b->angvel[1], b->angvel[2] };
		}
		bd.linearDamping = 0.0f;
		bd.angularDamping = 0.0f;
		bd.enableSleep = s->sleep != 0;
		bd.allowFastRotation = true; // no per-step rotation clamp (other engines have none)
		bd.enableContactRecycling = s->recycling != 0;
		b3BodyId id;
		if ( b->shape == 3 )
		{
			// No plane shape: a static box whose top face is the plane (normal = body +y)
			float h = s->planeHalfExtent;
			b3Vec3 down = b3RotateVector( bd.rotation, ( b3Vec3 ){ 0.0f, -1.0f, 0.0f } );
			bd.position = ( b3Pos ){ b->pos[0] + down.x, b->pos[1] + down.y, b->pos[2] + down.z };
			w->offset[3 * i] = -down.x, w->offset[3 * i + 1] = -down.y, w->offset[3 * i + 2] = -down.z;
			id = b3CreateBody( w->world, &bd );
			w->boxes[i] = b3MakeBoxHull( h, 1.0f, h );
		}
		else
		{
			id = b3CreateBody( w->world, &bd );
		}
		w->bodies[i] = id;
		w->dynamic[i] = !b->isStatic;

		b3ShapeDef sd = b3DefaultShapeDef();
		sd.baseMaterial.friction = b->friction;
		sd.baseMaterial.restitution = 0.0f;
		sd.density = 1.0f;
		sd.updateBodyMass = false; // exact scene mass properties are set below
		switch ( b->shape )
		{
			case 0:
				w->boxes[i] = b3MakeBoxHull( b->half[0], b->half[1], b->half[2] );
				// fallthrough
			case 3:
				b3CreateHullShape( id, &sd, &w->boxes[i].base );
				break;
			case 1:
			{
				b3Sphere sphere = { { 0.0f, 0.0f, 0.0f }, b->half[0] };
				b3CreateSphereShape( id, &sd, &sphere );
				break;
			}
			case 2:
				for ( uint32_t k = b->hullFirst; k < b->hullFirst + b->hullCount; ++k )
				{
					uint32_t first = hullTable[2 * k], count = hullTable[2 * k + 1];
					b3HullData* hull = b3CreateHull( (const b3Vec3*)( verts + 3 * first ), (int)count, B3_MAX_HULL_VERTICES );
					if ( hull == NULL )
					{
						report->failed += 1;
						snprintf( report->error, sizeof( report->error ), "body %d hull %u: b3CreateHull failed (%u points)", i, k, count );
						continue;
					}
					w->hulls[w->hullCount++] = hull;
					report->hullsIn += 1;
					report->verticesIn += (int)count;
					report->verticesOut += hull->vertexCount;
					report->hullsChanged += hull->vertexCount != (int)count;
					b3CreateHullShape( id, &sd, hull );
				}
				break;
		}
		if ( !b->isStatic )
		{
			b3MassData md = { 0 };
			md.mass = b->mass;
			md.center = ( b3Vec3 ){ 0.0f, 0.0f, 0.0f };
			md.inertia.cx = ( b3Vec3 ){ b->inertia[0], 0.0f, 0.0f };
			md.inertia.cy = ( b3Vec3 ){ 0.0f, b->inertia[1], 0.0f };
			md.inertia.cz = ( b3Vec3 ){ 0.0f, 0.0f, b->inertia[2] };
			b3Body_SetMassData( id, md );
		}
	}
	for ( int k = 0; k < nj; ++k )
	{
		const BxJoint* j = joints + k;
		b3SphericalJointDef jd = b3DefaultSphericalJointDef();
		jd.base.bodyIdA = w->bodies[j->a];
		jd.base.bodyIdB = w->bodies[j->b];
		jd.base.localFrameA = ( b3Transform ){ { j->anchorA[0], j->anchorA[1], j->anchorA[2] }, b3Quat_identity };
		jd.base.localFrameB = ( b3Transform ){ { j->anchorB[0], j->anchorB[1], j->anchorB[2] }, b3Quat_identity };
		jd.base.collideConnected = j->collide != 0;
		b3CreateSphericalJoint( w->world, &jd );
	}
	return w;
}

void bx_step( BxWorld* w )
{
	b3World_Step( w->world, w->dt, w->substeps );
}

// 13 floats per body: position, quaternion (x y z w), linear and angular velocity
void bx_read( BxWorld* w, float* out )
{
	for ( int i = 0; i < w->bodyCount; ++i )
	{
		b3BodyId id = w->bodies[i];
		b3Pos p = b3Body_GetPosition( id );
		b3Quat q = b3Body_GetRotation( id );
		b3Vec3 v = b3Body_GetLinearVelocity( id );
		b3Vec3 a = b3Body_GetAngularVelocity( id );
		float* o = out + 13 * i;
		o[0] = (float)p.x + w->offset[3 * i], o[1] = (float)p.y + w->offset[3 * i + 1], o[2] = (float)p.z + w->offset[3 * i + 2];
		o[3] = q.v.x, o[4] = q.v.y, o[5] = q.v.z, o[6] = q.s;
		o[7] = v.x, o[8] = v.y, o[9] = v.z;
		o[10] = a.x, o[11] = a.y, o[12] = a.z;
	}
}

typedef struct
{
	int32_t awake, contacts, awakeContacts, recycledContacts, islands, joints, bodies, tasks;
} BxCounters;

void bx_counters( BxWorld* w, BxCounters* c )
{
	b3Counters k = b3World_GetCounters( w->world );
	int awake = 0;
	for ( int i = 0; i < w->bodyCount; ++i )
	{
		awake += w->dynamic[i] && b3Body_IsAwake( w->bodies[i] );
	}
	c->awake = awake;
	c->contacts = k.contactCount;
	c->awakeContacts = k.awakeContactCount;
	c->recycledContacts = k.recycledContactCount;
	c->islands = k.islandCount;
	c->joints = k.jointCount;
	c->bodies = k.bodyCount;
	c->tasks = k.taskCount;
}

void bx_destroy( BxWorld* w )
{
	b3DestroyWorld( w->world );
	for ( int i = 0; i < w->hullCount; ++i ) b3DestroyHull( w->hulls[i] );
	free( w->hulls );
	free( w->boxes );
	free( w->bodies );
	free( w->offset );
	free( w->dynamic );
	free( w );
}

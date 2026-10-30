/* The smallest Rubble host: a crate dropped onto a floor, in Unreal units
 * (Z up, centimetres). CI compiles it against include/rubble.h, links it to
 * librubble_ffi and runs it; it exits 0 once the crate rests on the floor.
 *
 *   cc -std=c11 -I include examples/hello.c <lib dir>/librubble_ffi.a <system libs> -o hello
 *
 * engine/scripts/build-ffi.sh does this for each platform. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "rubble.h"

static void check(RbResult r, const char *what) {
    if (r != RB_OK) {
        fprintf(stderr, "%s failed (%d): %s\n", what, (int)r, rbGetLastError());
        exit(1);
    }
}

int main(void) {
    RbVersion v = rbGetVersion();
    printf("Rubble C ABI %u.%u\n", v.major, v.minor);

    RbWorldDef wd = rbDefaultWorldDef();
    wd.coordinates = RB_COORDINATES_UNREAL;
    wd.gravity = (RbVec3){0, 0, -981}; /* cm/s^2 */
    RbWorld *world = NULL;
    check(rbCreateWorld(&wd, &world), "rbCreateWorld");

    const RbTransform identity = {{0, 0, 0}, {0, 0, 0, 1}};
    RbShapeDef sd = rbDefaultShapeDef();

    /* A static floor whose top face is z = 0. */
    RbBodyDef fd = rbDefaultBodyDef();
    fd.transform.p = (RbVec3){0, 0, -50};
    RbBodyId floor_body;
    check(rbCreateBody(world, &fd, &floor_body), "rbCreateBody (floor)");
    RbShapeId floor_shape;
    check(rbCreateBoxShape(world, floor_body, &sd, (RbVec3){1000, 1000, 50}, identity,
                           &floor_shape),
          "rbCreateBoxShape (floor)");

    /* A 1 m crate 5 cm above the floor. */
    RbBodyDef bd = rbDefaultBodyDef();
    bd.type_ = RB_DYNAMIC_BODY;
    bd.transform.p = (RbVec3){0, 0, 55};
    bd.user_data = 42;
    RbBodyId crate_body;
    check(rbCreateBody(world, &bd, &crate_body), "rbCreateBody (crate)");
    sd.density = 0.0005f; /* kg/cm^3 */
    RbShapeId crate_shape;
    check(rbCreateBoxShape(world, crate_body, &sd, (RbVec3){50, 50, 50}, identity, &crate_shape),
          "rbCreateBoxShape (crate)");

    /* Two simulated seconds. Steps never wait; frames arrive a few steps
     * later, and each one lists only the bodies that moved. */
    RbTransform crate_pose = bd.transform;
    unsigned frames = 0;
    for (int step = 0; step < 240; ++step) {
        check(rbWorld_Step(world), "rbWorld_Step");
        RbFrame f;
        bool wait = step == 239; /* drain the last step */
        check(rbWorld_Poll(world, /*coalesce*/ true, wait, &f), "rbWorld_Poll");
        if (!f.has_frame) continue;
        ++frames;
        for (uint32_t i = 0; i < f.move_count; ++i) {
            if (f.moves[i].body == crate_body && f.moves[i].user_data == 42) {
                crate_pose = f.moves[i].transform;
            }
        }
    }

    printf("frames %u, crate at (%.2f, %.2f, %.2f) cm\n", frames, crate_pose.p.x, crate_pose.p.y,
           crate_pose.p.z);
    rbDestroyWorld(world);

    /* Resting on the floor: its centre half a metre up, less the contact
     * margin (about 1 cm) that resting contacts sit within. */
    if (frames == 0 || fabsf(crate_pose.p.z - 50.0f) > 1.5f || fabsf(crate_pose.p.x) > 1.0f) {
        fprintf(stderr, "the crate did not come to rest on the floor\n");
        return 1;
    }
    puts("ok");
    return 0;
}

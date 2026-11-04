/* The fracture lab through the C ABI: a scene pack's 2x2 tower block (its city's cars
 * and pedestrians in the streets) placed under a bond model, the block's volley queued,
 * the steps, and the stats every lab host shows. The same Rust the browser lab and the
 * native tests run (rubble::lab); the cross-host check compares their numbers.
 *
 *   lab <pack.json> [bonds] [steps]
 *
 * bonds: none, d6, threshold, load_path, impact_local, blast_gpu (default blast_gpu);
 * steps: default 480. Prints one line: steps=... shots=... bodies=... broken=... crushed=...
 *
 *   cc -std=c11 -I include examples/lab.c <lib dir>/librubble_ffi.a <system libs> -o lab */
/* MSVC: fopen and friends without the _s warnings (built with -WX). */
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rubble.h"

static void check(RbResult r, const char *what) {
    if (r != RB_OK) {
        fprintf(stderr, "%s failed (%d): %s\n", what, (int)r, rbGetLastError());
        exit(1);
    }
}

static RbLabBonds bonds_named(const char *name) {
    static const struct {
        const char *name;
        RbLabBonds bonds;
    } names[] = {
        {"none", RB_LAB_BONDS_NONE},           {"d6", RB_LAB_BONDS_D6},
        {"threshold", RB_LAB_BONDS_THRESHOLD}, {"load_path", RB_LAB_BONDS_LOAD_PATH},
        {"impact_local", RB_LAB_BONDS_IMPACT_LOCAL}, {"blast_gpu", RB_LAB_BONDS_BLAST_GPU},
    };
    for (size_t i = 0; i < sizeof names / sizeof names[0]; i++) {
        if (strcmp(names[i].name, name) == 0) return names[i].bonds;
    }
    fprintf(stderr, "unknown bonds %s\n", name);
    exit(2);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <pack.json> [bonds] [steps]\n", argv[0]);
        return 2;
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f) {
        perror(argv[1]);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long length = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *bytes = malloc((size_t)length);
    if (!bytes || fread(bytes, 1, (size_t)length, f) != (size_t)length) {
        fprintf(stderr, "could not read %s\n", argv[1]);
        return 1;
    }
    fclose(f);

    RbLabDef def = rbDefaultLabDef();
    def.source = RB_LAB_PACK;
    def.bytes = bytes;
    def.length = (size_t)length;
    def.arrangement = RB_LAB_TOWER_BLOCK;
    def.bonds = bonds_named(argc > 2 ? argv[2] : "blast_gpu");
    def.ccd = true;
    RbLab *lab = NULL;
    check(rbCreateLab(&def, &lab), "rbCreateLab");
    free(bytes);

    const RbVec3 zero = {0, 0, 0};
    check(rbLab_Queue(lab, RB_LAB_VOLLEY, zero, zero, 0, 0), "rbLab_Queue (volley)");
    unsigned steps = argc > 3 ? (unsigned)atoi(argv[3]) : 480;
    check(rbLab_Step(lab, steps), "rbLab_Step");

    RbLabStats s;
    check(rbLab_GetStats(lab, &s), "rbLab_GetStats");
    printf("steps=%llu shots=%u bodies=%u broken=%u crushed=%u bonds=%u chunks=%u gpu_step_ms=%.3f\n",
           (unsigned long long)s.steps, s.shots, s.bodies, s.bonds - s.bonds_intact, s.crushed,
           s.bonds, s.chunks, s.gpu_step_ms);
    rbDestroyLab(lab);
    return s.shots > 0 && s.bonds_intact < s.bonds ? 0 : 1;
}

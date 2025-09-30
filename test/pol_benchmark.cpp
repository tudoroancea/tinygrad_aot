#include "benchmark/benchmark.h"

extern void pol_2_10(double *__restrict__ out0_10, double *__restrict__ in0_20);
extern void pol2_10_2(double *__restrict__ out0_10,
                      double *__restrict__ in0_20);

#define no_optimize(ptr) asm volatile("" : : "r,m"(ptr[0]) : "memory");

static void pol_2_10(benchmark::State &state) {
  double out[10], in[20];

  for (auto _ : state) {
    pol_2_10(out, in);
    no_optimize(out);
  }
}
BENCHMARK(pol_2_10);

static void pol_10_2(benchmark::State &state) {
  double out[10], in[20];

  for (auto _ : state) {
    pol2_10_2(out, in);
    no_optimize(out);
  }
}
BENCHMARK(pol_10_2);

BENCHMARK_MAIN();

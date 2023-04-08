#include <stdio.h> 
#include <stdlib.h>
#include <math.h> 
#include <time.h>

#include "knn_v1.cuh"

inline void debugMode(){
  #ifndef ONLINE_JUDGE
  FILE *fp = freopen("input.txt", "r", stdin);
  // freopen("output.txt", "w", stdout);
  #endif //ONLINE_JUDGE
}

int main() {
  // #ifdef INPUT
    debugMode();    
  // #endif
  //Read total points to evaluate
  int n_points = read_number_of_points();
  #ifdef NDEBUG
    printf("n_points=%d\n", n_points);
  #endif
  //Read total k
  int f = read_k();
  #ifdef NDEBUG
    printf("k=%d\n", f);
  #endif

  GPUPoint *points = (GPUPoint*)malloc(n_points*sizeof(GPUPoint));
  for (int i = 0; i < n_points; i++){
      points[i] = read_gpu_point();
  }
  
  Point to_evaluate = read_point();
  char result =  wrapper_kSmallest(points, n_points, to_evaluate , f);
  
  printf("label=%c\n", result);

  free(points);
  return 0;
}
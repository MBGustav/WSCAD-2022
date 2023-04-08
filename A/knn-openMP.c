
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "knn.h"

void debugMode()
{
    #ifndef ONLINE_JUDGE
    FILE *fp = freopen("input.txt", "r", stdin);
    // freopen("output.txt", "w", stdout);
    #endif // ONLINE_JUDGE
}
char most_frequent(char *array, int k){
    char most_freq = array[0];
    // printf("most freq %c", most_freq);
    int most_freq_count = 1; 
    int curr_freq = 1;
    for(int i = 1; i < k; i++){

        if(array[i] != array[i-1]){
            if(curr_freq > most_freq_count){
                most_freq = array[i-1];
                most_freq_count = curr_freq;
            }
            curr_freq = 1;
        } 
        curr_freq++;

        if(i == k-1 && curr_freq > most_freq_count){
            most_freq = array[i-1];
            most_freq_count = curr_freq;
        }
    }
    return most_freq;
}

char knn_mp(Point *arr, int n , Point P, int k){
    int i, j, min_index;
    Point temp;
    char *result = (char*) malloc(sizeof(char)*k);

    #pragma omp parallel for private(i, j, min_index, temp) shared(arr, result)
    for (i = 0; i < k; i++) {
        min_index = i;
        for (j = i + 1; j < n; j++) {
            if(euclidean_distance_no_sqrt(arr[j], P) < euclidean_distance_no_sqrt(arr[min_index], P))
                min_index = j;
        }

        temp = arr[min_index];
        arr[min_index] = arr[i];
        arr[i] = temp;
    }
    
    //Write results --> smallest distances from "evaluate"
    #pragma omp barrier
    if (omp_get_thread_num() == 0) {
        for (i = 0; i < k; i++) {
            result[i] = arr[i].label;
        }
    }

    qsort(result, k, sizeof(char), compare_for_sort);

    char most_freq = most_frequent(result, k);
    return most_freq;
}

int main()
{
#ifdef INPUT
    debugMode();
#endif
    // Read total points to evaluate
    int n_points = read_number_of_points();
#ifdef NDEBUG
    printf("n_points=%d\n", n_points);
#endif
    // Read total k
    int f = read_k();
#ifdef NDEBUG
    printf("k=%d\n", f);
#endif
    Point *points = (Point *)malloc(n_points * sizeof(Point));
    for (int i = 0; i < n_points; i++)
    {
        points[i] = read_point();
        
    }

    Point to_evaluate = read_point_no_label();

    char result = knn_mp(points, n_points, to_evaluate, f);
    printf("label=%c\n",result);
    free(points);
    return 0;
}
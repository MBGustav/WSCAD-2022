#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include "knn.h"

#define SEED 777

int main(int argc, char **argv){
    char abc[] = "ABCDEFGHFHIJKLMNOPQRSTUVXZ";

    FILE *fp;
    int n_points, n_labels;
    if (argc <= 2) {
        printf ("ERROR: ./input-generator <n_points> <n_labels> \n");
        return 1;
    }

    fp = fopen( "input.txt" , "w" );

    n_points = atoi(argv[1]);
    n_labels = atoi(argv[2]);
    if(n_labels > 26){
        printf("Error: n_labels must be less than 26!\n");
        exit(1); 
    }

    fprintf(fp, "n_points=%d\n", n_points); 
    fprintf(fp, "k=%d\n", n_labels);

    
    Point temp, P;
    for(int i = 0; i < n_points; i++){
        temp.x  =(float) (rand() %100)*0.8; 
        temp.y  =(float) (rand() %100)*0.8; 
        temp.label  = abc[rand()%n_labels];
        fprintf(fp, "(%.2f,%.2f,%c)\n",temp.x,  temp.y,  temp.label);
    }

    P.x = (float) (rand() %100)*2.8;
    P.y = (float) (rand() %100)*1.8;

    fprintf(fp, "(%2f,%2f)\n",P.x,P.y);
    printf("Input Generated - n_points=%d, k=%d\n", n_points, n_labels);
    printf("P(x,y)=(%.2f,%.2f)\n", P.x, P.y);

    fclose(fp);
    return(0);

}
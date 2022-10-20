#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

typedef struct {
	long int *data;
	int length;
	long int total;
} bucket;

void sort(char *a, bucket *bucket) {
	int j, i, length;
	long int key;
	length = bucket->length;

	for (j = 1; j < bucket->total; j++) {
		key = bucket->data[j];
		i = j - 1;
		while (i >= 0 && strcmp(a + bucket->data[i] * length, a + key * length) > 0) {
			bucket->data[i + 1] = bucket->data[i];
			i--;
		}
		bucket->data[i + 1] = key;
	}
}


long int* bucket_sort(char *a, int length, long int size, int offset, int nbuckets) {

	long int i;
	bucket   *buckets = NULL,
             *b = NULL;
	long int *returns = NULL;

	// allocate memory
	returns = (long int *) malloc(sizeof(long int) * size);
	buckets = (bucket   *) malloc(sizeof(bucket) * nbuckets);


	for (i = 0; i < nbuckets; i++) {
		buckets[i].data = returns + i * size / nbuckets;
		buckets[i].length = length;
		buckets[i].total = 0;
	}

	// copy the keys to "buckets"
	for (i = 0; i < size; i++) {
		b = &buckets[*(a + i * length) - offset]; //mascara de deslocamento
		b->data[b->total++] = i;
	}

	// sort each "bucket"
	for (i = 0; i < nbuckets; i++)
		sort(a, &buckets[i]);

	return returns;
}

void load(char **vString, long int *N, int *l, int *os, int *nb){
   char *strings = NULL;
   int      len,
            offset,
            nbucket;
   long int n;

   if (!fscanf(stdin, "%d", &len)) {
      printf ("ERROR len\n");
      exit (1);
   }

   if (!fscanf(stdin, "%ld", &n)) {
      printf ("ERROR n\n");
      exit (1);
   }

   if (!fscanf(stdin, "%d", &offset)) {
      printf ("ERROR offset\n");
      exit (1);
   }

   if (!fscanf(stdin, "%d", &nbucket)) {
      printf ("ERROR nbucket\n");
      exit (1);
   }

   nbucket = nbucket + 1;
   len = len + 1;

   strings = (char*) malloc(n * len);

   for (int i = 0; i < n; i++){
      if (!fscanf(stdin, "%s", strings + (i * len))) {
      printf ("ERROR %d\n",i);
      exit (1);
      }
   }

   *vString = strings;
   *N = n;
   *l = len;
   *os = offset;
   *nb = nbucket;
}

void print(char *strings, long int *index, long int n, int len){

   for (int i = 0; i < n; i++){
      printf("%d -> %s\n", i, strings + (index[i] * len));
   }
}

void save(char *strings, long int *index, long int n, int len){
   for (int i = 0; i < n; i++){
      printf("%s\n", strings + (index[i] * len));
   }
}

int main(int ac, char **av) {

   char     *strings = NULL;
   long int N        = 0;
   long int *index   = NULL;
   int      len      = 0,
            save2File = 0,
            offset,
            nbuckets;

   load(&strings, &N, &len, &offset, &nbuckets);

   index = bucket_sort(strings, len, N, offset, nbuckets);

   save(strings, index, N, len);

   if (strings != NULL)
      free(strings);

   if (index != NULL)
      free(index);

   return EXIT_SUCCESS;
}

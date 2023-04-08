

typedef struct {
    float x;
    float y;
    char label;
} Point;


void on_error() {
    printf("Invalid input file.\n");
    exit(1);
}

void print_point(Point point){
    printf("(%.2f,%.2f,%c)\n",point.x, point.y, point.label);
}

int read_number_of_points(){
    int n;
    if(scanf(" n_points=%d \n", &n) != 1) on_error();
    return n;
}

int read_k(){
    int n;
    if(scanf(" k=%d \n", &n) != 1) on_error();
    return n;
}

Point read_point() {
    float x, y;
    char c;
    if (scanf("(%f,%f,%c)\n", &x, &y,&c) != 3)  on_error();
    Point point;
    point.x = x;
    point.y = y;
    point.label = c;
    return point;
}

Point read_point_no_label() {
    float x, y;
    if (fscanf(stdin," (%f ,%f) ", &x, &y) != 2)  on_error();
    Point point;
    point.x = x;
    point.y = y;

    return point;
}

float euclidean_distance_no_sqrt(Point a, Point b) {
    return ((b.x - a.x) * ((b.x - a.x)) + ((b.y - a.y) * (b.y - a.y)));
}

int compare_for_sort(const void *a, const void *b) {
  return *(char*)a - *(char*)b;
}


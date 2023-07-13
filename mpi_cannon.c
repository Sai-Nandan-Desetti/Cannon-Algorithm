/* Checkerboard implementation of Cannon's algorithm 
	for matrix multiplication.
   
   Author: Sai Nandan Desetti
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <mpi.h>

////////////////////////// MACRO Constants and typedefs /////////////////////////
#define MPI_TYPE MPI_INT

// for MPI_Cart_create()
#define REORDER 1

// for alloc_matrix()
#define SUCCESS 0
#define OUT_OF_MEMORY_ERROR 1

// for MPI_Cart_shift()
#define ROW_DIR 1
#define COL_DIR 0

typedef int Element_type;

////////////////////// FUNCTION PROTOTYPES //////////////////////
int get_size(MPI_Datatype);

void alloc_matrix(
   int      nrows,
   int      ncols,
   size_t   element_size,
   Element_type **matrix_storage,
   Element_type ***matrix,
   int      *errvalue
);

void read_and_distribute_2dblock_matrix(
   char           *filename,
   Element_type   ***M,
   Element_type   **M_storage,
   MPI_Datatype   dtype,
   int            *nrows,
   int            *ncols,
   int            *errval,
   MPI_Comm       cart_comm  
);

void print_vector(int *buffer, int n, MPI_Datatype dtype, FILE *file);

void collect_and_print_2dblock_matrix(
   Element_type   **a,
   MPI_Datatype   dtype,
   int            m,
   int            n,
   MPI_Comm       grid_comm
);


/////////////////////////////////////// main() ///////////////////////////////
int main(int argc, char *argv[]) {

   /* Variable declarations */
   int id;
   int p, p_root;
   double p_root_d;
   int grid_id;
   int grid_size[2];
   int grid_coords[2];
   int periodic[2];
   int error;
   int source, dest;
   int nrows, ncols;
   int nlocal_rows, nlocal_cols;
   int *bufferA, *bufferB;
   Element_type **A;
   Element_type *A_storage;
   Element_type **B;
   Element_type *B_storage;
   Element_type **C;
   Element_type *C_storage;   
   MPI_Comm grid_comm;   
   MPI_Status status;
   /* End of declarations */


   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &id);
   MPI_Comm_size(MPI_COMM_WORLD, &p);

   p_root_d = sqrt((double) p);
   p_root = p_root_d;
   // if p is not a perfect square, we do not implement Cannon's algorithm
   assert(p_root == p_root_d);
   
   // the size of the grid is (p_root, p_root)
   grid_size[0] = p_root;
   grid_size[1] = p_root;   
   
   periodic[0] = 1;
   periodic[1] = 1;
   
   // create a Cartesian grid of processes for grid A
   MPI_Cart_create(MPI_COMM_WORLD, 2, grid_size, periodic, REORDER, &grid_comm);
   // rank of a process in the grid
   MPI_Comm_rank(grid_comm, &grid_id);
   // coordinates of a process in the grid
   MPI_Cart_coords(grid_comm, grid_id, 2, grid_coords);   

   read_and_distribute_2dblock_matrix(argv[1], &A, &A_storage, MPI_TYPE, &nrows, &ncols, &error, grid_comm);      
   if (error != SUCCESS)
      MPI_Abort(MPI_COMM_WORLD, MPI_ERR_IO);         

   read_and_distribute_2dblock_matrix(argv[2], &B, &B_storage, MPI_TYPE, &nrows, &ncols, &error, grid_comm);      
   if (error != SUCCESS)
      MPI_Abort(MPI_COMM_WORLD, MPI_ERR_IO);         

   nlocal_rows = nrows/grid_size[0];
   nlocal_cols = ncols/grid_size[1];     
   
   // allocate memory for each local block of the output matrix C
   alloc_matrix(nlocal_rows, nlocal_cols, get_size(MPI_TYPE), &C_storage, &C, &error);
   if (error != SUCCESS)
      MPI_Abort(MPI_COMM_WORLD, MPI_ERR_IO);  
   
   // initialize each local block matrix of C to all zeros
   for (int i = 0; i < nlocal_rows; i++)
      for (int j = 0; j < nlocal_cols; j++)
         C[i][j] = 0;

   // buffer arrays used to store the shifted matrix elements
   bufferA = A_storage;   
   bufferB = B_storage;

   // initial shift of i positions to the left for the ith row of A's meshgrid,
   // and by i positions upward for the ith column of B's meshgrid
   MPI_Cart_shift(grid_comm, ROW_DIR, grid_coords[0], &source, &dest);
   MPI_Sendrecv_replace(bufferA, nlocal_rows*nlocal_cols, MPI_TYPE, dest, 1, source, 1, grid_comm, &status);   
   
   MPI_Cart_shift(grid_comm, COL_DIR, grid_coords[1], &source, &dest);
   MPI_Sendrecv_replace(bufferB, nlocal_rows*nlocal_cols, MPI_TYPE, dest, 1, source, 1, grid_comm, &status);
   
   // shift by one step grid_size times
   for (int step = 0; step < grid_size[0]; step++) {      

      // local block multiplication 
      // these local blocks will be collected later and printed in order
      for (int i = 0; i < nlocal_rows; i++) {      
         for (int j = 0; j < nlocal_cols; j++) {         
            for (int k = 0; k < nlocal_cols; k++)
               C[i][j] += bufferA[i*nlocal_rows+k] * bufferB[k*nlocal_rows+j];               
         }         
      }
      // shift by one
      // -row for A
      // -column for B
      MPI_Cart_shift(grid_comm, ROW_DIR, 1, &source, &dest);
      MPI_Sendrecv_replace(bufferA, nlocal_rows*nlocal_cols, MPI_TYPE, dest, 1, source, 1, grid_comm, &status);

      MPI_Cart_shift(grid_comm, COL_DIR, 1, &source, &dest);
      MPI_Sendrecv_replace(bufferB, nlocal_rows*nlocal_cols, MPI_TYPE, dest, 1, source, 1, grid_comm, &status);
   }      

   // print the results
   if (grid_id == 0) printf("Matrix A:\n");   
   collect_and_print_2dblock_matrix(A, MPI_TYPE, nrows, ncols, grid_comm);

   if (grid_id == 0) printf("Matrix B:\n"); 
   collect_and_print_2dblock_matrix(B, MPI_TYPE, nrows, ncols, grid_comm); 

   if (grid_id == 0) printf("Matrix C:\n");
   collect_and_print_2dblock_matrix(C, MPI_TYPE, nrows, ncols, grid_comm);


   MPI_Finalize();
   return 0;
}

//////////////////////////////////////////// FUNCTIONS //////////////////////////////////////
int get_size(MPI_Datatype datatype) {
   if (datatype == MPI_INT)
      return sizeof(int);
   else
      return -1;
}

void alloc_matrix(
   // function parameters
   int      nrows,
   int      ncols,
   size_t   element_size,
   Element_type **matrix_storage,
   Element_type ***matrix,
   int      *errval
)
{
   void *ptr_to_row_in_storage;
   void **matrix_row_start;  

   *matrix_storage = malloc(nrows * ncols * element_size);
   if (*matrix_storage == NULL) {
      *errval = OUT_OF_MEMORY_ERROR;      
      return;
   }

   *matrix = malloc (nrows * sizeof(void *));
   if (*matrix == NULL) {
      *errval = OUT_OF_MEMORY_ERROR;      
      return;
   }

   matrix_row_start = (void *) &(*matrix[0]);

   ptr_to_row_in_storage = (void *) *matrix_storage;

   for (int i = 0; i < nrows; i++ ) {
      *matrix_row_start = (void *) ptr_to_row_in_storage;
      matrix_row_start++;
      ptr_to_row_in_storage += ncols * element_size;
   }

   *errval = SUCCESS;
}

void read_and_distribute_2dblock_matrix(
   // function parameters
   char           *filename,
   Element_type   ***M,
   Element_type   **M_storage,
   MPI_Datatype   dtype,
   int            *nrows,
   int            *ncols,
   int            *errval,
   MPI_Comm       cart_comm  
) 
{
   // variable declarations   
   int mpi_initialized;
   int grid_id;   
   FILE *file;
   size_t element_size;
   int grid_size[2];
   int grid_periodic[2];
   int grid_coords[2];
   int nlocal_rows;
   int nlocal_cols;      
   int *buffer;
   int block_coords[2];
   int dest_id;
   Element_type *source_address;
   Element_type *dest_address;
   MPI_Status status;

   // check if MPI_Init was called already
   // if not, set appropriate error value and return
   MPI_Initialized(&mpi_initialized);
   if (!mpi_initialized) {
      *errval = -1;
      return;
   }

   // get the rank of each process   
   MPI_Comm_rank(cart_comm, &grid_id);   

   element_size = get_size(dtype);
   if (element_size == -1) {
      *errval = -1;
      return;
   }

   // get the number of rows and columns in the cartesian grid...
   if (grid_id == 0) {
      file = fopen(filename, "r");
      if (file == NULL) {
         *nrows = 0;         
      }
      else {
         fscanf(file, "%d", nrows);
      }
      *ncols = *nrows;
   }
   // ...and broadcast to all the processes the number of rows...
   MPI_Bcast(nrows, 1, MPI_INT, 0, MPI_COMM_WORLD);
   if (*nrows == 0) {
      *errval = -1;
      return;
   }
   // ...and number of columns
   MPI_Bcast(ncols, 1, MPI_INT, 0, MPI_COMM_WORLD);
   
   // each process needs the topology info to compute 
   // the appropriate number of rows and cols in its sub-block
   MPI_Cart_get(cart_comm, 2, grid_size, grid_periodic, grid_coords);
   
   // each sub-block must be of size n/sqrt(p), where n is the dimension of the input matrix
   // this is why n must be a multiple of sqrt(p)
   nlocal_rows = *nrows/grid_size[0];
   nlocal_cols = *ncols/grid_size[1];

   // allocate space for the sub-block in *M_storage; it will be pointed to by *M
   alloc_matrix(nlocal_rows, nlocal_cols, element_size, M_storage, M, errval);
   if (*errval != SUCCESS) {
      fprintf(stderr, "Couldn't allocate memory for a sub-block.\n");
      MPI_Abort(cart_comm, *errval);
   }
      
   if (grid_id == 0) {
      buffer = (int *) malloc(*ncols * element_size);
      if (buffer == NULL) {
         fprintf(stderr, "Couldn't allocate space for buffer.\n");
         MPI_Abort(cart_comm, *errval);
      }         
   }
   for (int i = 0; i < grid_size[0]; i++) {
      block_coords[0] = i;
      for (int j = 0; j < nlocal_rows; j++) {
         if (grid_id == 0) {
            for (int b = 0; b < *ncols; b++) {
               fscanf(file, "%d", &buffer[b]);
            }
         }
         for (int k = 0; k < grid_size[1]; k++) {
            block_coords[1] = k;
            MPI_Cart_rank(cart_comm, block_coords, &dest_id);

            if (grid_id == 0) {
               source_address = buffer + k * nlocal_cols;

               if (dest_id == 0) {
                  dest_address = (*M)[j];
                  memcpy(dest_address, source_address, nlocal_cols * element_size);
               }
               else {                                                 
                  MPI_Send(source_address, nlocal_cols, dtype, dest_id, 0, cart_comm);
               }
            }
            else if (grid_id == dest_id) {
               MPI_Recv((*M)[j], nlocal_cols, dtype, 0, 0, cart_comm, &status);
            }
         }
      }
   }
   if (grid_id == 0) {
      fclose(file);
      free(buffer);
   }  
   
   *errval = SUCCESS;
   return;
}

void print_vector(int *buffer, int n, MPI_Datatype dtype, FILE *file) {

   if (dtype == MPI_INT) {
      for (int i = 0; i < n; i++) {
         fprintf(file, "%d\t", buffer[i]);
      }
   }
   fprintf(file, "\n");
}

void collect_and_print_2dblock_matrix(
   Element_type   **a,
   MPI_Datatype   dtype,
   int            m,
   int            n,
   MPI_Comm       grid_comm
)
{
   int *buffer;
   int coords[2];
   int element_size;   
   int grid_coords[2];
   int grid_id;
   int grid_periodic[2];
   int grid_size[2];
   void *laddr;
   int local_rows, local_cols;   
   int src;
   MPI_Status status;

   MPI_Comm_rank(grid_comm, &grid_id);
   element_size = get_size(dtype);

   MPI_Cart_get(grid_comm, 2, grid_size, grid_periodic, grid_coords);

   local_rows = m/grid_size[0];
   local_cols = n/grid_size[1];

   if (grid_id == 0) {
      buffer = (int *) malloc (n * element_size);
   }
   for (int i = 0; i < grid_size[0]; i++) {
      coords[0] = i;

      for (int j = 0; j < local_rows; j++) {
         if (grid_id == 0) {
            for (int k = 0; k < grid_size[1]; k++) {
               coords[1] = k;
               MPI_Cart_rank(grid_comm, coords, &src);               
               laddr = buffer + k * local_cols;

               if (src == 0)
                  memcpy(laddr, a[j], local_cols * element_size);               
               else 
                  MPI_Recv(laddr, local_cols, dtype, src, 0, grid_comm, &status);
            }
            print_vector(buffer, n, dtype, stdout);
         }
         else if (grid_coords [0] == i) {
            MPI_Send (a[j], local_cols, dtype, 0, 0, grid_comm);
         }
      }
   }
   if (grid_id == 0) {
      free(buffer);
   }
}
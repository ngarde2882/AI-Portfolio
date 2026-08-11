// 
// Computes the product of two matrices: C = A * B
// using Strassen's algorithm
//
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <new>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace std::chrono;

#define MAX_MATRIX_SIZE	65536
#define TOL 1.0e-12

#define DEBUG 1

// Global variables
int matrix_size; 
int leaf_matrix_size;

// Define Matrix

class Matrix {
    public:
        int  nrows;		// number of rows
        int  ncols;		// number of columns 
        double **elements;	// Matrix elements

	static Matrix strassens_product(Matrix&, Matrix&);
	static Matrix standard_product(Matrix&, Matrix&);
	static Matrix addition(Matrix&, Matrix&);
	static Matrix subtraction(Matrix&, Matrix&);
	static int compare_matrix(Matrix&, Matrix&); 

	Matrix extract_submatrix(int, int, int, int); 
	void update_submatrix(Matrix&, int, int, int, int); 
	void initialize_matrix(double);
//	void free_memory();
	void print_matrix();

	static void matrix_error(int); 

	Matrix(int,int); 
	~Matrix();

    Matrix(const Matrix&); // copy constructor
    Matrix& operator=(const Matrix&); // = overload
    Matrix(Matrix&&) noexcept; // move constructor
    Matrix& operator=(Matrix&&) noexcept; // move = overload

    private:
	double *array;
};

// Strassen's matrix product 
// - return C = A * B
// - implemented only for square matrices where nrows=ncols
//
Matrix Matrix::strassens_product(Matrix& A, Matrix& B) {

    int n = A.nrows;
    if (A.nrows != A.ncols) Matrix::matrix_error(500); 
    if (B.nrows != B.ncols) Matrix::matrix_error(501); 
    if (A.ncols != B.nrows) Matrix::matrix_error(502); // Matrix not square

    if (n <= leaf_matrix_size) {

	Matrix C = Matrix::standard_product(A, B); 
	return C;

    } else {
        // Extract blocks of A: A11, A12, A21, A22
        Matrix A11 = A.extract_submatrix(0, n/2-1, 0, n/2-1);
        Matrix A12 = A.extract_submatrix(0, n/2-1, n/2, n-1);
        Matrix A21 = A.extract_submatrix(n/2, n-1, 0, n/2-1);
        Matrix A22 = A.extract_submatrix(n/2, n-1, n/2, n-1);
        
        // Extract blocks of B: B11, B12, B21, B22
        Matrix B11 = B.extract_submatrix(0, n/2-1, 0, n/2-1);
        Matrix B12 = B.extract_submatrix(0, n/2-1, n/2, n-1);
        Matrix B21 = B.extract_submatrix(n/2, n-1, 0, n/2-1);
        Matrix B22 = B.extract_submatrix(n/2, n-1, n/2, n-1);
        
        Matrix M1(n/2,n/2), M2(n/2,n/2), M3(n/2,n/2), M4(n/2,n/2), M5(n/2,n/2), M6(n/2,n/2), M7(n/2,n/2);
            // Compute products M1, M2, ..., M7
            Matrix M1a = Matrix::addition(A11,A22); 
            Matrix M1b = Matrix::addition(B11,B22);
            M1 = Matrix::strassens_product(M1a, M1b); 
            Matrix M2a = Matrix::addition(A21,A22);
            M2 = Matrix::strassens_product(M2a, B11); 
            Matrix M3b = Matrix::subtraction(B12,B22);
            M3 = Matrix::strassens_product(A11, M3b); 
            Matrix M4b = Matrix::subtraction(B21,B11);
            M4 = Matrix::strassens_product( A22, M4b);
            Matrix M5a = Matrix::addition(A11,A12); 
            M5 = Matrix::strassens_product(M5a, B22);
            Matrix M6a = Matrix::subtraction(A21,A11); 
            Matrix M6b = Matrix::addition(B11,B12);
            M6 = Matrix::strassens_product(M6a, M6b);
            Matrix M7a = Matrix::subtraction(A12,A22); 
            Matrix M7b = Matrix::addition(B21,B22);
            M7 = Matrix::strassens_product(M7a, M7b);
	// Compute blocks of C: C11, C12, C21, C22
	Matrix C11a = Matrix::addition(M1,M4);
	Matrix C11b = Matrix::subtraction(M7,M5);
        Matrix C11 = Matrix::addition(C11a, C11b);

        Matrix C12 = Matrix::addition(M3,M5);
        Matrix C21 = Matrix::addition(M2,M4);

	Matrix C22a = Matrix::subtraction(M1,M2);
	Matrix C22b = Matrix::addition(M3,M6);
        Matrix C22 = Matrix::addition(C22a, C22b);

	// Create the product matrix C
	Matrix C(n,n); 
        C.update_submatrix(C11, 0, n/2-1, 0, n/2-1);
        C.update_submatrix(C12, 0, n/2-1, n/2, n-1);
        C.update_submatrix(C21, n/2, n-1, 0, n/2-1);
        C.update_submatrix(C22, n/2, n-1, n/2, n-1);

        return C;
    }

}
 
// Standard matrix product
// - return C = A * B
Matrix Matrix::standard_product(Matrix& A, Matrix& B) {
    if (A.ncols != B.nrows) matrix_error(5); 
    Matrix C(A.nrows,B.ncols); 
    for (int i = 0; i < C.nrows; i++) {
        for (int j = 0; j < C.ncols; j++) {
	    C.elements[i][j] = 0.0;
	    for (int k = 0; k < A.ncols; k++) 
	        C.elements[i][j] += A.elements[i][k]*B.elements[k][j];
	}
    } 
    return C;
}

// Standard matrix addition
// - return C = A + B
Matrix Matrix::addition(Matrix& A, Matrix& B) {
    if (A.nrows != B.nrows) A.matrix_error(8); 
    if (A.ncols != B.ncols) A.matrix_error(9); 
    Matrix C(A.nrows,A.ncols);  
    for (int i = 0; i < C.nrows; i++) {
        for (int j = 0; j < C.ncols; j++) {
	    C.elements[i][j] = A.elements[i][j]+B.elements[i][j];
	}
    } 
    return C;
}

// Standard matrix subtraction
// - return C = A - B
Matrix Matrix::subtraction(Matrix& A, Matrix& B) {
    if (A.nrows != B.nrows) A.matrix_error(88); 
    if (A.ncols != B.ncols) A.matrix_error(98); 
    Matrix C(A.nrows,A.ncols);  
    for (int i = 0; i < C.nrows; i++) {
        for (int j = 0; j < C.ncols; j++) {
	    C.elements[i][j] = A.elements[i][j]-B.elements[i][j];
	}
    } 
    return C;
}

// Compare if matrix is identical to another by checking if 
// their elements are identical within specified tolerance
int Matrix::compare_matrix(Matrix& A, Matrix& B) {
    int error = 0;
    if (A.nrows != B.nrows) return error; 
    if (A.ncols != B.ncols) return error; 
    for (int i = 0; i < A.nrows; i++) {
        for (int j = 0; j < A.ncols; j++) {
            if (fabs(A.elements[i][j] - B.elements[i][j]) > TOL) error = 1;
	}
    }
    return error;
}

// Extract submatrix of A
// - return S = A[row_first:row_last][col_first:col_last]
Matrix Matrix::extract_submatrix(int row_first, int row_last, 
			  	 int col_first, int col_last) {
    Matrix S(row_last-row_first+1, col_last-col_first+1); 
    for (int i = 0; i < S.nrows; i++) {
        for (int j = 0; j < S.ncols; j++) {
	    S.elements[i][j] = elements[row_first+i][col_first+j];
	}
    } 
    return S;
}

// Update submatrix of A with matrix S
// - Copy S into A[row_first:row_last][col_first:col_last]
void Matrix::update_submatrix(Matrix& S, int row_first, int row_last, 
					int col_first, int col_last) {
    if (S.nrows != (row_last-row_first+1)) S.matrix_error(100);
    if (S.ncols != (col_last-col_first+1)) S.matrix_error(101);
    if (nrows < row_last) S.matrix_error(102); 
    if (ncols < col_last) S.matrix_error(103); 
    for (int i = 0; i < S.nrows; i++) {
        for (int j = 0; j < S.ncols; j++) {
	    elements[row_first+i][col_first+j] = S.elements[i][j];
	}
    } 
}

// Initialize matrix
// - for testing purpose only
void Matrix::initialize_matrix(double factor) {
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            elements[i][j]= i + factor*j;
	}
    }
}

// Print matrix
void Matrix::print_matrix() {
    printf("\n... Printing matrix ... \n");
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
   	    printf(" %8.4f",elements[i][j]);
	}
	printf("\n");
    }
}

// Generic error
void Matrix::matrix_error(int error_number) {
    printf("Error encountered: %d ... aborting\n", error_number);  
    exit(0);
}

// Create new matrix
Matrix::Matrix(int num_rows, int num_cols) {
    nrows = num_rows;
    ncols = num_cols;
    elements = new double *[nrows];
    array = new double[nrows*ncols];
    for (int i = 0; i < nrows; i++) elements[i] = &(array[i*ncols]);
}

// Destroy matrix - free dynamically allocated memory
Matrix::~Matrix(){
	delete [] elements; 
	delete [] array; 
//	if (elements != nullptr) { delete [] elements; elements = nullptr; }
//	if (array != nullptr) { delete [] array; array = nullptr; }
}

// Copy matrix - copy data from one matrix to another
Matrix::Matrix(const Matrix& other) {
    nrows = other.nrows;
    ncols = other.ncols;
    elements = new double*[nrows];
    array = new double[nrows * ncols];
    for (int i = 0; i < nrows; i++) {
        elements[i] = &(array[i * ncols]);
        for (int j = 0; j < ncols; j++)
            array[i * ncols + j] = other.array[i * ncols + j];
    }
}

// = overload
Matrix& Matrix::operator=(const Matrix& other) {
    if (this == &other) return *this;  // self-assignment guard

    // free existing memory
    delete[] elements;
    delete[] array;

    nrows = other.nrows;
    ncols = other.ncols;
    elements = new double*[nrows];
    array = new double[nrows * ncols];
    for (int i = 0; i < nrows; i++) {
        elements[i] = &(array[i * ncols]);
        for (int j = 0; j < ncols; j++)
            array[i * ncols + j] = other.array[i * ncols + j];
    }
    return *this;
}

// Move matrix - change the owner of a matrix (when a deep copy isn't necessary)
Matrix::Matrix(Matrix&& other) noexcept {
    nrows = other.nrows;
    ncols = other.ncols;
    elements = other.elements;
    array = other.array;
    other.elements = nullptr;
    other.array = nullptr;
    other.nrows = 0;
    other.ncols = 0;
}

// move = overload
Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this == &other) return *this;

    delete[] elements;
    delete[] array;

    nrows = other.nrows;
    ncols = other.ncols;
    elements = other.elements;
    array = other.array;

    other.elements = nullptr;
    other.array = nullptr;
    other.nrows = 0;
    other.ncols = 0;

    return *this;
}

// =======================================================================
int main(int argc, char *argv[]) {
    time_t start, end;
    double standard_time, strassens_time;

    // Read input, validate
    if (argc != 3) {
        printf("Need two integers as input \n"); 
        printf("Use: <executable_name> <log_2(matrix_size)> <log_2(leaf_matrix_size)>\n"); 
        exit(0);
    }   
    int k = atoi(argv[argc-2]);
    matrix_size = (1 << k); 
    if (matrix_size > MAX_MATRIX_SIZE) {
        printf("Maximum matrix size allowed: %d.\n", MAX_MATRIX_SIZE);
        exit(0);
    };  
    int q = atoi(argv[argc-1]);
    leaf_matrix_size = (1 << q); 
    if (leaf_matrix_size > matrix_size) {
        printf("Leaf matrix size too large, setting to matrix size ...\n");
        leaf_matrix_size = matrix_size;
    };  

    // Initialize matrices A and B 
    Matrix A(matrix_size,matrix_size); A.initialize_matrix(1.0); 
    Matrix B(matrix_size,matrix_size); B.initialize_matrix(-1.0); 

    // ----------------------------------
    // Strassen's matrix multiplication
    start = omp_get_wtime();
    Matrix C = Matrix::strassens_product(A,B); 
    strassens_time = omp_get_wtime() - start;

    printf("Matrix size = %d, Leaf matrix size = %d, Strassen's (s) = %8.4f s,",
            matrix_size, leaf_matrix_size, strassens_time);

    // ----------------------------------
    // Standard matrix multiplication

    // Compute Cstd = A*B - standard matrix multiplication
    if (k < 11) {
    	start = omp_get_wtime();
	    Matrix Cstd = Matrix::standard_product(A,B); 
    	time(&end); 
    	standard_time = omp_get_wtime() - start;

	    printf(" Standard = %8.4f s,", standard_time);

	    int error = Matrix::compare_matrix(C,Cstd);
	    printf(" Error = %d\n", error);

    	if (error != 0) {
	        printf("Houston, we have a problem!\n");
	    }
	} else {
        printf(" Standard = not computed for large matrices \n");
	}
}


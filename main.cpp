#include <cmath>
#include "mpi.h"
#include <iostream>
#include <vector>

constexpr auto t = 0.0000001;
constexpr auto e_pow_2 = 0.0000001 * 0.0000001;
constexpr auto RANK_ROOT = 0;
constexpr auto N = 200;
constexpr auto BREAK = 0;
constexpr auto CONTINUE = 1;

void Matrix_Multiply(const double *buf, int lines, const double *x, double *tmp) {
    for (int i = 0; i < lines; i++) {
        double sum = 0;
        for (int j = 0; j < N; j++) {
            sum += buf[i * N + j] * x[j];
        }
        tmp[i] = sum;
    }
}

std::vector<int> Get_lines_count(int size, int *offset) {
    std::vector<int> lines_count(size);
    int value = N / size;
    for (int i = 0; i < size; i++) {
        lines_count[i] = value;
    }

    int index = 0;
    int cur = N % size;
    while (cur - index++ > 0) {
        lines_count[index - 1]++;
    }
    int cur_offset = 0;
    for (int i = 0; i < size; i++) {
        offset[i] = cur_offset;
        cur_offset += lines_count[i];
    }
    return lines_count;
}

double Norma(const double *vector, int size) {
    double tmp = 0;
    for (int i = 0; i < size; i++) {
        tmp += vector[i] * vector[i];
    }
    return tmp;
}

void Vector_Difference(const double * a_1, const double *a_2, int size, int offset, double *tmp) {
    for (int i = offset; i < size + offset; i++) {
        tmp[i - offset] = a_1[i - offset] - a_2[i];
    }
}

void Vector_Multiply_Const(const double *a_1, double value, int size, double *tmp) {
    for (int i = 0; i < size; i++) {
        tmp[i] = a_1[i] * value;
    }
}

void Print_Vector(double *vector) {
    for (int i = 0; i < N; i++) {
        std::cout << vector[i] << ' ';
    }
    std::cout << '\n';
}

void Print_Vector(const int *vector) {
    for (int i = 0; i < N; i++) {
        std::cout << vector[i] << ' ';
    }
    std::cout << '\n';
}

void Print_Matrix(double *vector, int length, int width) {
    for (int i = 0; i < length; i++) {
        for (int j = 0; j < width; j++) {
            std::cout << vector[i * N + j] << ' ';
        }
        std::cout << '\n';
    }
}

void Print_Matrix(const int *vector, int length, int width) {
    for (int i = 0; i < length; i++) {
        for (int j = 0; j < width; j++) {
            std::cout << vector[i * N + j] << ' ';
        }
        std::cout << '\n';
    }
}

void Run(int size, int rank) {
    std::vector<int> offset(size);

    auto lines_count = Get_lines_count(size, offset.data());

    std::vector<double> b(N);
    std::vector<double> x(N);
    std::vector<double> A_buf(lines_count[rank] * N);

    for (int i = 0; i < N; i++) {
        b[i] = N + 1;
        x[i] = 0;
    }
    if (rank == RANK_ROOT) {
        std::vector<double> A((long long )N * (long long)N);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i * N + j] = (i == j) ? 2.0 : 1.0;
            }
        }
        for (int other_rank = 1; other_rank < size; other_rank++) {
            MPI_Send(A.data() + N * offset[other_rank], lines_count[other_rank] * N, MPI_DOUBLE,
                     other_rank, 1, MPI_COMM_WORLD);
        }

        for (int i = 0; i < lines_count[rank]; i++) {
            for (int j = 0; j < N; j++) {
                A_buf[i * N + j] = A[i * N + j];
            }
        }
    } else {
        MPI_Recv(A_buf.data(), lines_count[rank] * N, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    double norma_b = Norma(b.data(), N);

    std::vector<double> tmp(lines_count[rank]);
    std::vector<double> new_x_part(lines_count[rank]);
    std::vector<double> multiply_tmp_const(lines_count[rank]);

    while (true) {
        if (rank == RANK_ROOT) {
            for (int other_rank = 1; other_rank < size; other_rank++) {
                MPI_Send(x.data(), N, MPI_DOUBLE, other_rank, CONTINUE, MPI_COMM_WORLD);
            }
            Matrix_Multiply(A_buf.data(), lines_count[RANK_ROOT], x.data(), tmp.data());
            Vector_Difference(tmp.data(), b.data(), lines_count[rank], offset[RANK_ROOT], tmp.data());
            
            double norma = Norma(tmp.data(), lines_count[RANK_ROOT]);
            
            std::vector<double> epsilon_part(1);
            for (int other_rank = 1; other_rank < size; other_rank++) {
                MPI_Recv(epsilon_part.data(), 1, MPI_DOUBLE, other_rank, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                norma += epsilon_part[0];
            }

            Vector_Multiply_Const(tmp.data(), t, lines_count[rank], multiply_tmp_const.data());
            Vector_Difference(x.data(), multiply_tmp_const.data(), lines_count[rank], offset[rank], new_x_part.data());

            MPI_Gatherv(new_x_part.data(), lines_count[rank], MPI_DOUBLE, x.data(), lines_count.data(), offset.data(),
                        MPI_DOUBLE, 0, MPI_COMM_WORLD);

            if (norma / norma_b < e_pow_2) {
                /*std::cout << "x:";
                Print_Vector(x.data());*/
                for (int other_rank = 1; other_rank < size; other_rank++) {
                    MPI_Send(x.data(), N, MPI_DOUBLE, other_rank, BREAK, MPI_COMM_WORLD);
                }
                break;
            }
            

        } else {
            MPI_Status status;
            MPI_Recv(x.data(), N, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (status.MPI_TAG == BREAK) {
                break;
            }

            Matrix_Multiply(A_buf.data(), lines_count[rank], x.data(), tmp.data());
            Vector_Difference(tmp.data(), b.data(), lines_count[rank], offset[rank], tmp.data());

            double norma = Norma(tmp.data(), lines_count[rank]);

            MPI_Send(&norma, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

            Vector_Multiply_Const(tmp.data(), t, lines_count[rank], multiply_tmp_const.data());
            Vector_Difference(multiply_tmp_const.data(), x.data(), lines_count[rank], offset[rank], new_x_part.data());

            Vector_Multiply_Const(new_x_part.data(), -1, lines_count[rank], new_x_part.data());

            MPI_Gatherv(new_x_part.data(), lines_count[rank], MPI_DOUBLE, x.data(), lines_count.data(), offset.data(),
                        MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);
        }
    }
}

int main(int argc, char **argv) {
    int size, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const double start_time_s = MPI_Wtime();
    Run(size, rank);
    const double end_time_s = MPI_Wtime();
    std::cout << "rank " << rank << ": " << end_time_s - start_time_s << '\n';
    MPI_Finalize();
    return 0;
}

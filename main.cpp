#include <cmath>
#include "mpi.h"
#include <iostream>
#include <vector>

constexpr auto t = 0.00001;
constexpr auto e_pow_2 = 0.00001 * 0.00001;
constexpr auto RANK_ROOT = 0;
constexpr auto N = 15000;
constexpr auto BREAK = 0;
constexpr auto CONTINUE = 1;

void MatrixMultiply(const double *buf, int lines, const double *x, double *tmp) {
    for (int i = 0; i < lines; i++) {
        double sum = 0;
        for (int j = 0; j < N; j++) {
            sum += buf[i * N + j] * x[j];
        }
        tmp[i] = sum;
    }
}

std::vector<int> GetlinesCount(int size, int *offset) {
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

void VectorDifference(const double * a_1, const double *a_2, int size, int offset, double *tmp) {
    for (int i = offset; i < size + offset; i++) {
        tmp[i - offset] = a_1[i - offset] - a_2[i];
    }
}

void VectorMultiplyConst(const double *a_1, double value, int size, double *tmp) {
    for (int i = 0; i < size; i++) {
        tmp[i] = a_1[i] * value;
    }
}

double GetSummaryNorm(int size, double rank_root_norm) {
    double cur_norma = rank_root_norm;

    std::vector<double> epsilon_part(1);
    for (int other_rank = 1; other_rank < size; other_rank++) {
        MPI_Recv(epsilon_part.data(), 1, MPI_DOUBLE, other_rank, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cur_norma += epsilon_part[0];
    }

    return cur_norma;
}

bool NormaCompare(double norma, int size, double norma_b, const std::vector<double>& x) {
    if (norma / norma_b < e_pow_2) {
        /*for (auto i: x)
            std::cout << i << " ";*/
        for (int other_rank = 1; other_rank < size; other_rank++) {
            MPI_Send(x.data(), N, MPI_DOUBLE, other_rank, BREAK, MPI_COMM_WORLD);
        }
        return true;
    }
    return false;
}

void SendA(const std::vector<int>& offset, const std::vector<int>& lines_count, int rank, std::vector<double>& A_buf) {
    std::vector<int> lines_count_mult_N(lines_count.begin(), lines_count.end());
    for (int & i : lines_count_mult_N) {
        i *= N;
    }

    std::vector<int> offset_mult_N(offset.begin(), offset.end());
    for (int & i : offset_mult_N) {
        i *= N;
    }
    if (rank == RANK_ROOT) {
        std::vector<double> A((long long) N * (long long) N);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i * N + j] = (i == j) ? 2.0 : 1.0;
            }
        }
        MPI_Scatterv(A.data(), lines_count_mult_N.data(), offset_mult_N.data(), MPI_DOUBLE, A_buf.data(),
                     lines_count_mult_N[rank], MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(nullptr, lines_count_mult_N.data(), offset_mult_N.data(), MPI_DOUBLE, A_buf.data(),
                     lines_count_mult_N[rank], MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);
    }
}

void Run(int size, int rank) {
    std::vector<int> offset(size);

    auto lines_count = GetlinesCount(size, offset.data());

    std::vector<double> b(N);
    std::vector<double> x(N);
    std::vector<double> A_buf(lines_count[rank] * N);

    for (int i = 0; i < N; i++) {
        b[i] = N + 1;
        x[i] = 0;
    }

    SendA(offset, lines_count, rank, A_buf);

    double norma_b = Norma(b.data(), N);

    std::vector<double> tmp(lines_count[rank]);
    std::vector<double> new_x_part(lines_count[rank]);
    std::vector<double> multiply_tmp_const(lines_count[rank]);

    while (true) {
        if (rank == RANK_ROOT) {
            for (int other_rank = 1; other_rank < size; other_rank++) {
                MPI_Send(x.data(), N, MPI_DOUBLE, other_rank, CONTINUE, MPI_COMM_WORLD);
            }
            MatrixMultiply(A_buf.data(), lines_count[RANK_ROOT], x.data(), tmp.data());
            VectorDifference(tmp.data(), b.data(), lines_count[rank], offset[RANK_ROOT], tmp.data());

            double norma = GetSummaryNorm(size, Norma(tmp.data(), lines_count[RANK_ROOT]));

            VectorMultiplyConst(tmp.data(), t, lines_count[rank], multiply_tmp_const.data());
            VectorDifference(x.data(), multiply_tmp_const.data(), lines_count[rank], offset[rank], new_x_part.data());

            MPI_Gatherv(new_x_part.data(), lines_count[rank], MPI_DOUBLE, x.data(), lines_count.data(), offset.data(),
                        MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);

            if (NormaCompare(norma, size, norma_b, x)) {
                break;
            }
        } else {
            MPI_Status status;
            MPI_Recv(x.data(), N, MPI_DOUBLE, RANK_ROOT, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (status.MPI_TAG == BREAK) {
                break;
            }

            MatrixMultiply(A_buf.data(), lines_count[rank], x.data(), tmp.data());
            VectorDifference(tmp.data(), b.data(), lines_count[rank], offset[rank], tmp.data());

            double norma = Norma(tmp.data(), lines_count[rank]);
            VectorMultiplyConst(tmp.data(), t, lines_count[rank], multiply_tmp_const.data());
            VectorDifference(multiply_tmp_const.data(), x.data(), lines_count[rank], offset[rank], new_x_part.data());

            VectorMultiplyConst(new_x_part.data(), -1, lines_count[rank], new_x_part.data());
            MPI_Send(&norma, 1, MPI_DOUBLE, RANK_ROOT, CONTINUE, MPI_COMM_WORLD);

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

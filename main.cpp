#include <mpi.h>
#include <pthread.h>
#include <cstdlib>
#include <iostream>
#include <list>
#include <cmath>
 
using namespace std;
 
const int ITERATIONS = 5;
const int PLANNED_TASKS = 150;
const int RequestWork = 1;
const int ReceiveWork = 2;
 
pthread_mutex_t mutex;
std::list<int> *taskList;
int tasksGotFromOtherProcs = 0;
int tasksGivenOtherProcs = 0;
 
void* manage(void *arg) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
 
    int requestFromProcNo = 0;
    while (true) {
        MPI_Recv(&requestFromProcNo, 1, MPI_INT, MPI_ANY_SOURCE, RequestWork, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        if (requestFromProcNo >= size)
            break;
 
        int val = 0;
        pthread_mutex_lock(&mutex);
        if (!taskList->empty()) {
            val = taskList->back();
            taskList->pop_back();
            tasksGivenOtherProcs++;
        }
        pthread_mutex_unlock(&mutex);
 
        MPI_Send(&val, 1, MPI_INT, requestFromProcNo, ReceiveWork, MPI_COMM_WORLD);
    }
}
 
list<int> *createTaskList(int iteration) {
    auto *l = new list<int>;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
 
    const int L = 100000;
    for (int i = 0; i < PLANNED_TASKS; ++i) {
        l->push_back(L * abs(50 - i) * abs(rank - (iteration % size)));
    }
    return l;
}
 
void requestWork() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
 
    int val = 0;
    for (int i = 0; i < size; i++) {
        if (i != rank)
            MPI_Send(&rank, 1, MPI_INT, i, RequestWork, MPI_COMM_WORLD);
    }
 
    for (int i = 0; i < size; i++) {
        if (i != rank) {
            MPI_Recv(&val, 1, MPI_INT, i, ReceiveWork, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (val != 0) {
                pthread_mutex_lock(&mutex);
                taskList->push_back(val);
                tasksGotFromOtherProcs++;
                pthread_mutex_unlock(&mutex);
            }
        }
    }
}
 
void printStats(double procTime, int iteration, double sum) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
 
    int root = 0;
    double minProcTime;
    double maxProcTime;
    double delta;
    double disbalanceProportion;
 
    MPI_Allreduce(&procTime, &maxProcTime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&procTime, &minProcTime, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
 
    delta = maxProcTime - minProcTime;
    disbalanceProportion = delta / maxProcTime * 100;
 
    double* times = new double[size]();
    int* allTasksGotFromOtherProcs = new int[size]();
    int* allTasksGivenOtherProcs = new int[size]();
    int* allSum = new int[size]();
 
    MPI_Gather(&procTime, 1, MPI_DOUBLE, times, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);
    MPI_Gather(&tasksGotFromOtherProcs, 1, MPI_INT, allTasksGotFromOtherProcs,
               1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Gather(&tasksGivenOtherProcs, 1, MPI_INT, allTasksGivenOtherProcs,
               1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Gather(&sum, 1, MPI_DOUBLE, allSum, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);
 
    if (rank == root) {
        cout << "iteration: " << iteration <<", delta: " << delta << ", disbalance proportion: "
             << disbalanceProportion << "%\n";
        for (int i = 0; i < size; i++) {
            cout << "rank: " << i << ", time: " << times[i] << ", tasks: "
                 << PLANNED_TASKS + allTasksGotFromOtherProcs[i] - allTasksGivenOtherProcs[i] << ", sum: "
                 << allSum[i] << "\n";
        }
    }
 
    delete[] times;
    delete[] allTasksGotFromOtherProcs;
    delete[] allTasksGivenOtherProcs;
    delete[] allSum;
}
 
int main(int argc, char** argv) {
    int rank, size, provided = 0;
 
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE) {
        cout << "MPI issues\n" << endl;
        return 0;
    }
 
    pthread_mutex_init(&mutex, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 
    pthread_t managerThread;
    pthread_create(&managerThread, NULL, manage, NULL);
 
    for (int i = 0; i < ITERATIONS; i++) {
        taskList = createTaskList(i);
        double sum = 0, startTime = MPI_Wtime();
        int task;
        while (!taskList->empty()) {
            if (!taskList->empty()) {
                pthread_mutex_lock(&mutex);
                    task = taskList->front();
                    taskList->pop_front();
                pthread_mutex_unlock(&mutex);
            }
 
            for (int k = 0; k < task; k++)
                sum += sin(k);
 
            if (taskList->empty())
                requestWork();
        }
        double endTime = MPI_Wtime();
 
        MPI_Barrier(MPI_COMM_WORLD);
        printStats(endTime - startTime, i, sum);
        delete taskList;
    }
 
    MPI_Send(&size, 1, MPI_INT, rank, RequestWork, MPI_COMM_WORLD);
    pthread_join(managerThread, NULL);
    pthread_mutex_destroy(&mutex);
    MPI_Finalize();
    return 0;
}

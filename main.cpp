#include <vector>
#include <chrono>
#include <execution>
#include <iostream>
#include <armadillo>


class MyTask
{
    public:
        MyTask(int size) : size_(size) {};
        double get_result();

    protected:
        int size_;
};

double MyTask::get_result()
{
    arma::mat big_mat = arma::randu<arma::mat>(size_,size_);
    return arma::cond(big_mat);
}


int main(int argc, char *argv[])
{
    int loop_size = 1000;
    int prob_size = 200;

    std::cout << std::endl;

    // Set up the vector of tasks 
    std::cout << "setting up problem" << std::endl;
    std::vector<MyTask> task_vec(loop_size, MyTask(prob_size));
    auto element_func = [](MyTask task) {return task.get_result();};
    std::cout << std::endl;

    // Run tasks sequentially
    std::cout << "running sequential computation " << std::endl; 
    std::vector<double> seq_result_vec(loop_size);
    auto t_start = std::chrono::high_resolution_clock::now();
    std::transform(
            std::execution::seq, 
            std::begin(task_vec), 
            std::end(task_vec), 
            std::begin(seq_result_vec), 
            element_func
            );
    auto t_stop = std::chrono::high_resolution_clock::now();
    std::cout << "t = " << std::chrono::duration<double>(t_stop - t_start).count();
    std::cout << std::endl << std::endl;

    // Run tasks in parallel
    std::cout << "running parallel computation " << std::endl; 
    std::vector<double> par_result_vec(loop_size);
    t_start = std::chrono::high_resolution_clock::now();
    std::transform(
            std::execution::par, 
            std::begin(task_vec), 
            std::end(task_vec), 
            std::begin(par_result_vec), 
            element_func
            );
    t_stop = std::chrono::high_resolution_clock::now();
    std::cout << "t = " << std::chrono::duration<double>(t_stop - t_start).count();
    std::cout << std::endl << std::endl;

    return 0;
}

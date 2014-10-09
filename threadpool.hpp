/*
 *Copyright (c) 2012 Jakob Progsch
 *
 *This software is provided 'as-is', without any express or implied
 *warranty. In no event will the authors be held liable for any damages
 *arising from the use of this software.
 *
 *Permission is granted to anyone to use this software for any purpose,
 *including commercial applications, and to alter it and redistribute it
 *freely, subject to the following restrictions:
 *
 *  1. The origin of this software must not be misrepresented; you must not
 *  claim that you wrote the original software. If you use this software
 *  in a product, an acknowledgment in the product documentation would be
 *  appreciated but is not required.
 *
 *  2. Altered source versions must be plainly marked as such, and must not be
 *  misrepresented as being the original software.

 *  3. This notice may not be removed or altered from any source
 *  distribution.
 */

#ifndef THREAD_POOL_HPP
#define THREAD_POOL_HPP

#include <stack>
#include <queue>
#include <mutex>
#include <memory>
#include <thread>
#include <vector>
#include <future>
#include <stdexcept>
#include <functional>
#include <type_traits>
#include <condition_variable>

#include <iostream>

typedef std::function<void(void)> task_t;

class threadpool {
private:
    std::vector<std::thread> mWorkers;
    std::queue<task_t> mTasks;

    std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> isActive;
public:
    threadpool (size_t numThreads = std::thread::hardware_concurrency() ) : isActive(true){
        for(size_t i = 0 ; i < numThreads; i++)
            mWorkers.emplace_back(std::thread(&threadpool::scheduler_loop,this));
    }

private:
    void scheduler_loop(){
        while(1){
            std::unique_lock<std::mutex> lock(this->queue_mutex);
            while(this->mTasks.empty()){
                if( !this->isActive.load() ) return;
                this->condition.wait(lock);
            }
            std::function<void()> lNextTask = this->mTasks.front();
            this->mTasks.pop();
            lock.unlock();
            lNextTask();
        }
    }

public:

    threadpool(threadpool& to_copy) = delete; //Probably wouldn't and shouldn't copy this.
    void operator =(threadpool& to_copy) = delete;

    template<class F, class... Args> //Below is the return type...Yes it is ridiculous, but it works. Enabled if policy_tpye IS NOT PRIORITY
    std::future< typename std::result_of<F(Args...)>::type >
    enqueue(F&& f, Args&&... args){
        typedef typename std::result_of<F(Args...)>::type return_type;
        if ( ! isActive.load() ) // Don't allow enqueueing after stopping the pool
            throw std::runtime_error("enqueue on stopped ThreadPool");

        auto task = std::make_shared<std::packaged_task<return_type()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...) );

        std::future<return_type> result = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            mTasks.push([task](){ (*task)(); });
        }
        condition.notify_one();
        return result;
    }

    void close(){
        this->isActive.store(false);
        condition.notify_all();
        for(std::thread& t : mWorkers)
            t.join();
    }

    ~threadpool(){
        this->close();
    }
};
#endif

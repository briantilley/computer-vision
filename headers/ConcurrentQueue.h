//
// Copyright (c) 2013 Juan Palacios juan.palacios.puyana@gmail.com
// Subject to the BSD 2-Clause License
// - see < http://opensource.org/licenses/BSD-2-Clause>
//

#ifndef CONCURRENT_QUEUE_
#define CONCURRENT_QUEUE_

#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

template <typename T>
class ConcurrentQueue
{
private:
	const bool m_blank = false; // a "blank" queue is not meant to be used

public:

	T pop() 
	{
		std::unique_lock<std::mutex> mlock(mutex_);
		while (queue_.empty())
		{
			cond_.wait(mlock);
		}
		auto val = queue_.front();
		queue_.pop();
		return val;
	}

	void pop(T& item)
	{
		std::unique_lock<std::mutex> mlock(mutex_);
		while (queue_.empty())
		{
			cond_.wait(mlock);
		}
		item = queue_.front();
		queue_.pop();
	}

	void push(const T& item)
	{
		std::unique_lock<std::mutex> mlock(mutex_);
		queue_.push(item);
		mlock.unlock();
		cond_.notify_one();
	}

	// Brian Tilley briantilley97@gmail.com github.com/briantilley
	bool empty()
	{
		bool returnState;
		std::unique_lock<std::mutex> mlock(mutex_);
		returnState = queue_.empty();
		mlock.unlock();
		return returnState;
	}

	ConcurrentQueue()=default;
	ConcurrentQueue(bool blank): m_blank(true) {  }
	ConcurrentQueue(const ConcurrentQueue&) = delete;            // disable copying
	ConcurrentQueue& operator=(const ConcurrentQueue&) = delete; // disable assignment

	operator bool() const { return !m_blank; }
	
private:
	std::queue<T> queue_;
	std::mutex mutex_;
	std::condition_variable cond_;
};

#endif

#include "headers/CodedFrame.h"

// static member variable definition
std::map<const byte*, int> CodedFrame::allocations;

// copy raw data and wrap in object
CodedFrame::CodedFrame(const byte* _data, unsigned _length)
{
	// store size and allocate space
	length = _length;
	
	/*
	data = new byte[length];

	// C-style copy is OK because data is only raw bytes
	memcpy(reinterpret_cast<void*>(data), reinterpret_cast<const void*>(_data), length);
	*/

	// look for pointer
	// if it's already registered, increase its count
	// else, register the pointer with one associated instance
	allocationIterator = allocations.find(_data);
	if(allocationIterator != allocations.end())
		allocationIterator->second++;
	else
	{
		allocations.insert(std::pair<const byte*, int>(_data, 1));
		allocationIterator = allocations.find(_data);
	}
}
	
// copy constructor (same behavior as above)
CodedFrame::CodedFrame(const CodedFrame& toCopy)
{
	// store size and allocate space
	length = toCopy.size();

	/*
	data = new byte[length];

	// C-style copy is OK because data is only raw bytes
	memcpy(reinterpret_cast<void*>(data), reinterpret_cast<const void*>(toCopy.raw_data()), length);
	*/

	allocationIterator = toCopy.allocationIterator;
	allocationIterator->second++;
}

// copy assignment
void CodedFrame::operator=(const CodedFrame& right)
{
	/*
	// clear existing data
	delete [] data;
	*/

	// store size and allocate space
	length = right.size();
	
	/*
	data = new byte[length];

	// C-style copy is OK because data is only raw bytes
	memcpy(reinterpret_cast<void*>(data), reinterpret_cast<const void*>(right.raw_data()), length);
	*/

	allocationIterator = right.allocationIterator;
	allocationIterator->second++;
}
	
// free copy of data
CodedFrame::~CodedFrame()
{
	// one less instance pointing to memory location
	allocationIterator->second--;

	// if there's no instance tied to allocation,
	// deallocate and remove from map
	if(0 >= allocationIterator->second)
	{
		delete [] allocationIterator->first;
		allocations.erase(allocationIterator);
	}
}
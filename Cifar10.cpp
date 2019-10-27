#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
//#include "rc10.hpp"

void print_im(char* data) {
	for (unsigned chan = 0; chan < 3; ++chan) {
		std::cout << "New channel : \n";
		for (unsigned i = 0; i < 1024; ++i) {
			std::cout << +(uint8_t)data[1024*chan + i] << ' ';
			if ((i + 1) % 32 == 0)
				std::cout << '\n';
		}
		std::cout << '\n';
	}
	std::cout << std::flush;
}

int main() {
	constexpr unsigned N_IM = 1;
	std::ifstream file("C:/Users/Arthur/Desktop/CPP/Project1/Datasets/CIFAR-10/data_batch_1.bin", std::ios::binary);
	
	if (file.is_open()) {
		std::unique_ptr<char[]> buffer(new char[3073]);
		file.read(buffer.get(), 3073);
		file.read(buffer.get(), 3073);
		std::cout << "Label : " /*<< std::hex */<< +(uint8_t)buffer[0] << '\n';
		print_im(&buffer[1]);
	} else {
		std::cerr << "Failed to open file." << std::endl;
	}
	//auto dataset = cifar::read_dataset();
	//std::cout << dataset.training_labels[0];
	return 0;
}

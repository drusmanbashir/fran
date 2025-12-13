// #include <iostream>
// #include <torch/torch.h>
// #include <fstream>   
//
// std::vector<char> get_the_bytes(std::string filename) {
//     std::ifstream input(filename, std::ios::binary);
//     std::vector<char> bytes(
//         (std::istreambuf_iterator<char>(input)),
//         (std::istreambuf_iterator<char>()));
//
//     input.close();
//     return bytes;
// }
//
// int main()
// {
//     std::vector<char> f = get_the_bytes("/tmp/pt_tensor.pt");
//     torch::IValue x = torch::pickle_load(f);
//     torch::Tensor my_tensor = x.toTensor();
//     std::cout << "[cpp] my_tensor: " << my_tensor << std::endl;
//
//     return 0;
// }

#include <torch/torch.h>
#include <fstream>
#include <iostream>
#include <vector>

int main() {
    std::ifstream in("/home/ub/code/fran/fran/cpp/files/sample_tensor.pt", std::ios::binary);
    std::vector<char> data((std::istreambuf_iterator<char>(in)),
                            std::istreambuf_iterator<char>());

    torch::IValue iv = torch::pickle_load(data);
    torch::Tensor t = iv.toTensor();

    std::cout << t << "\n";
}

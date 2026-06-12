#pragma once
#include <string>
#include <memory>

class BlackwellEngine {
public:
    struct Impl; 
    
    explicit BlackwellEngine(const std::string& index_path, size_t max_seq_len = 2048);
    ~BlackwellEngine();

    int forward(int token_id, int pos, float temperature = 0.6f, float top_p = 0.9f);
    float forward_eval(int token_id, int pos, int target_token_id);

    Impl* get_impl() const { return pImpl.get(); }

private:
    std::unique_ptr<Impl> pImpl;
};
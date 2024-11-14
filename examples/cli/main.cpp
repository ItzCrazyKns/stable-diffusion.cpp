#include <stdio.h>
#include <string>
#include <vector>

// #include "preprocessing.hpp"
#include "flux.hpp"
#include "stable-diffusion.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_STATIC
#include "stb_image_resize.h"

#include "nlohmann/json.hpp"
#include <httplib.h>

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include <random>
#include <string>
#include <vector>

enum SDMode {
    TXT2IMG,
};

struct SDParams {
    int n_threads = -1;
    SDMode mode   = TXT2IMG;
    std::string model_path;
    std::string clip_l_path;
    std::string clip_g_path;
    std::string t5xxl_path;
    std::string diffusion_model_path;
    std::string vae_path;
    std::string taesd_path;
    std::string esrgan_path;
    std::string controlnet_path;
    std::string embeddings_path;
    std::string stacked_id_embeddings_path;
    std::string input_id_images_path;
    sd_type_t wtype = SD_TYPE_COUNT;
    std::string lora_model_dir;
    std::string output_path = "output.png";
    std::string input_path;
    std::string control_image_path;

    std::string prompt;
    std::string negative_prompt;
    float min_cfg     = 1.0f;
    float cfg_scale   = 7.0f;
    float guidance    = 3.5f;
    float style_ratio = 20.f;
    int clip_skip     = -1;  // <= 0 represents unspecified
    int width         = 512;
    int height        = 512;
    int batch_count   = 1;

    int video_frames         = 6;
    int motion_bucket_id     = 127;
    int fps                  = 6;
    float augmentation_level = 0.f;

    sample_method_t sample_method = EULER_A;
    schedule_t schedule           = DEFAULT;
    int sample_steps              = 20;
    float strength                = 0.75f;
    float control_strength        = 0.9f;
    rng_type_t rng_type           = CUDA_RNG;
    int64_t seed                  = 42;
    bool verbose                  = false;
    bool vae_tiling               = false;
    bool control_net_cpu          = false;
    bool normalize_input          = false;
    bool clip_on_cpu              = false;
    bool vae_on_cpu               = false;
    bool canny_preprocess         = false;
    bool color                    = false;
    int upscale_repeats           = 1;
};

std::string get_image_params(SDParams params, int64_t seed) {
    std::string parameter_string = params.prompt + "\n";
    parameter_string += "Steps: " + std::to_string(params.sample_steps) + ", ";
    parameter_string += "CFG scale: " + std::to_string(params.cfg_scale) + ", ";
    parameter_string += "Seed: " + std::to_string(seed) + ", ";
    parameter_string += "Size: " + std::to_string(params.width) + "x" + std::to_string(params.height) + ", ";
    return parameter_string;
}

void generate_image(const SDParams& params) {
    sd_ctx_t* sd_ctx = new_sd_ctx(params.model_path.c_str(),
                                  params.clip_l_path.c_str(),
                                  params.clip_g_path.c_str(),
                                  params.t5xxl_path.c_str(),
                                  params.diffusion_model_path.c_str(),
                                  params.vae_path.c_str(),
                                  params.taesd_path.c_str(),
                                  params.controlnet_path.c_str(),
                                  params.lora_model_dir.c_str(),
                                  params.embeddings_path.c_str(),
                                  params.stacked_id_embeddings_path.c_str(),
                                  true,
                                  params.vae_tiling,
                                  true,
                                  params.n_threads,
                                  params.wtype,
                                  params.rng_type,
                                  params.schedule,
                                  params.clip_on_cpu,
                                  params.control_net_cpu,
                                  params.vae_on_cpu);

    if (sd_ctx == NULL) {
        printf("Context initialization failed\n");
        return;
    }

    sd_image_t* results = txt2img(sd_ctx,
                                  params.prompt.c_str(),
                                  "", 0, params.cfg_scale, 0.0f,
                                  params.width, params.height, params.sample_method,
                                  params.sample_steps, params.seed, 1, nullptr, 0.9f, 20.0f, false, "");

    if (results == NULL) {
        printf("Image generation failed\n");
        free_sd_ctx(sd_ctx);
        return;
    }

    stbi_write_png(params.output_path.c_str(), results->width, results->height, results->channel, results->data, 0);
    printf("Image saved to '%s'\n", params.output_path.c_str());
    free(results->data);
    free(results);
    free_sd_ctx(sd_ctx);
}

void handle_request(const httplib::Request& req, httplib::Response& res) {
    nlohmann::json request_json = nlohmann::json::parse(req.body);
    SDParams params;
    params.prompt = request_json["prompt"].get<std::string>();
    params.width = request_json.contains("width") ? request_json["width"].get<int>() : 512;
    params.height = request_json.contains("height") ? request_json["height"].get<int>() : 512;
    params.cfg_scale = request_json.contains("cfg_scale") ? request_json["cfg_scale"].get<float>() : 7.0f;
    params.sample_steps = request_json.contains("sample_steps") ? request_json["sample_steps"].get<int>() : 20;
    params.seed = request_json.contains("seed") ? request_json["seed"].get<int64_t>() : 42;

    params.model_path = request_json.contains("model_path") ? request_json["model_path"].get<std::string>() : "../models/sd-v1-4.ckpt";

/* Log params in console */

    std::cout << "Prompt: " << params.prompt << std::endl;
    std::cout << "Width: " << params.width << std::endl;
    std::cout << "Height: " << params.height << std::endl;
    std::cout << "CFG Scale: " << params.cfg_scale << std::endl;
    std::cout << "Sample Steps: " << params.sample_steps << std::endl;
    std::cout << "Seed: " << params.seed << std::endl;
    std::cout << "Model Path: " << params.model_path << std::endl;

    generate_image(params);

    nlohmann::json response_json;
    response_json["status"] = "success";
    response_json["image_path"] = params.output_path;
    res.set_content(response_json.dump(), "application/json");
}

int main() {
    httplib::Server server;
    server.Post("/generate-image", handle_request);
    server.listen("0.0.0.0", 8080);
}
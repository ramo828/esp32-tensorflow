/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
... (Lisenziya mətni qısaldılmışdır) ...
==============================================================================*/

// TensorFlow Lite Micro üçün lazım olan kitabxanalar
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "constants.h"
// DİQQƏT: 'model.h' faylı layihənin yanında olmalı və ya model datası 
// (g_model) bu koda daxil edilməlidir.
#include "model.h" 

// Arduino uyğunluğu üçün qlobal dəyişənlər
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

// Modelin ölçüsünə görə bu sahəni artırmaq lazım ola bilər (Məsələn: 10000)
constexpr int kTensorArenaSize = 2000;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

void setup() {
  // Serial rabitəni başladırıq
  Serial.begin(9600);
  while (!Serial); // Serial portun hazır olmasını gözlə (Xüsusilə daxili USB-li kartlar üçün)
  
  Serial.println("TensorFlow Lite Micro Başladılır...");

  // Modeli yüklə
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("Model versiyası uyğun deyil! Dəstəklənən: ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    return;
  }

  // Yalnız bizə lazım olan əməliyyatları (Qatları) çağırırıq
  static tflite::MicroMutableOpResolver<1> resolver;
  if (resolver.AddFullyConnected() != kTfLiteOk) {
    Serial.println("FullyConnected qatı əlavə edilə bilmədi!");
    return;
  }

  // Interpreter (Tərcüməçi) qurulumu
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Tensor arenadan yaddaş ayrılması
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() uğursuz oldu!");
    return;
  }

  // Giriş və çıxış pointerlərini al
  input = interpreter->input(0);
  output = interpreter->output(0);

  inference_count = 0;
  Serial.println("Qurulum tamamlandı, təxminlər başlayır...");
}

void loop() {
  // X dəyərini hesabla
  float position = static_cast<float>(inference_count) /
                   static_cast<float>(kInferencesPerCycle);
  float x = position * kXrange;

  // Girişi modelin istədiyi kimi Kvantlaşdır (Float -> Int8)
  int8_t x_quantized = x / input->params.scale + input->params.zero_point;
  input->data.int8[0] = x_quantized;

  // Modeli işə sal (Təxmin et)
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.print("Təxmin yerinə yetirilə bilmədi! X: ");
    Serial.println(x);
    return;
  }

  // Çıxışı al və Kvantlaşdırmanı ləğv et (Int8 -> Float)
  int8_t y_quantized = output->data.int8[0];
  float y = (y_quantized - output->params.zero_point) * output->params.scale;

  // Orijinal HandleOutput funksiyasının əvəzinə birbaşa Serial Plotter-ə uyğun çıxış:
  Serial.print("X_Deyeri:");
  Serial.print(x);
  Serial.print(",");
  Serial.print("Y_Tehmini:");
  Serial.println(y);

  // Dövrü irəlilət və ehtiyac varsa sıfırla
  inference_count += 1;
  if (inference_count >= kInferencesPerCycle) {
    inference_count = 0;
  }

  delay(10); // Çıxışları rahat oxumaq üçün yüngül bir gecikmə
}

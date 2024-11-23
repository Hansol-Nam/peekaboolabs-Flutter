import UIKit
import Flutter
import CoreML
import Vision

@main
@objc class AppDelegate: FlutterAppDelegate {
    private var emotionModel: CoremlModelTestFloat16?

    override func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
    ) -> Bool {
        // CoreML 모델 로드
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndNeuralEngine
            self.emotionModel = try CoremlModelTestFloat16(configuration: config)
        } catch {
            fatalError("모델을 로드할 수 없습니다: \(error)")
        }

        let controller: FlutterViewController = window?.rootViewController as! FlutterViewController
        let emotionChannel = FlutterMethodChannel(name: "com.peekaboolabs.flutter_app3/emotion",
                                                  binaryMessenger: controller.binaryMessenger)

        emotionChannel.setMethodCallHandler { [weak self] (call, result) in
            guard let self = self else { return }

            if call.method == "getEmotion" {
                if let args = call.arguments as? [String: Any],
                   let faceBytes = args["faceBytes"] as? FlutterStandardTypedData {
                    self.predictEmotion(faceBytes: faceBytes.data, result: result)
                } else {
                    result(FlutterError(code: "INVALID_ARGUMENTS",
                                        message: "Face bytes are missing or invalid",
                                        details: nil))
                }
            } else {
                result(FlutterMethodNotImplemented)
            }
        }

        GeneratedPluginRegistrant.register(with: self)
        return super.application(application, didFinishLaunchingWithOptions: launchOptions)
    }

    private func predictEmotion(faceBytes: Data, result: @escaping FlutterResult) {
        guard let model = self.emotionModel else {
            result(FlutterError(code: "MODEL_NOT_LOADED",
                                message: "Emotion model is not loaded",
                                details: nil))
            return
        }

        // 얼굴 이미지 데이터를 UIImage로 변환
        guard let uiImage = UIImage(data: faceBytes) else {
            result(FlutterError(code: "INVALID_IMAGE",
                                message: "Unable to convert face bytes to UIImage",
                                details: nil))
            return
        }

        // UIImage를 MLMultiArray로 변환
        guard let multiArray = uiImage.toMLMultiArray(width: 224, height: 224) else {
            result(FlutterError(code: "MULTIARRAY_ERROR",
                                message: "Unable to convert UIImage to MLMultiArray",
                                details: nil))
            return
        }

        // CoreML 모델 예측
        do {
            let prediction = try model.prediction(x_1: multiArray)
            let scores = prediction.linear_72ShapedArray.scalars // [Float] 배열

            // 감정 리스트
            let emotionsList = ["sad", "disgust", "angry", "neutral", "fear", "surprise", "happy"]

            // 최고 점수의 인덱스 찾기
            if let maxScore = scores.max(),
               let maxIndex = scores.firstIndex(of: maxScore),
               maxIndex < emotionsList.count {
                let detectedEmotion = emotionsList[maxIndex]
                result(detectedEmotion)
            } else {
                result("알 수 없음")
            }
        } catch {
            result(FlutterError(code: "PREDICTION_ERROR",
                                message: "Error during prediction: \(error.localizedDescription)",
                                details: nil))
        }
    }
}

// UIImage를 MLMultiArray로 변환하는 확장
extension UIImage {
    func toMLMultiArray(width: Int = 224, height: Int = 224) -> MLMultiArray? {
        // 1. 이미지 리사이즈
        UIGraphicsBeginImageContextWithOptions(CGSize(width: width, height: height), true, 1.0)
        self.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let resizedImage = UIGraphicsGetImageFromCurrentImageContext() else { return nil }
        UIGraphicsEndImageContext()
        
        // 2. 픽셀 데이터 추출
        guard let cgImage = resizedImage.cgImage else { return nil }
        guard let pixelData = cgImage.dataProvider?.data else { return nil }
        let data: UnsafePointer<UInt8> = CFDataGetBytePtr(pixelData)
        
        // 3. MLMultiArray 생성 (1x3x224x224)
        guard let multiArray = try? MLMultiArray(shape: [1, 3, NSNumber(value: height), NSNumber(value: width)], dataType: .float32) else { return nil }
        
        // 4. 픽셀 데이터를 MLMultiArray에 채우기
        for y in 0..<height {
            for x in 0..<width {
                let pixelIndex = (y * width + x) * 4 // RGBA
                let r = Float(data[pixelIndex]) / 255.0
                let g = Float(data[pixelIndex + 1]) / 255.0
                let b = Float(data[pixelIndex + 2]) / 255.0
                
                multiArray[[0, 0, y, x] as [NSNumber]] = NSNumber(value: r) // R 채널
                multiArray[[0, 1, y, x] as [NSNumber]] = NSNumber(value: g) // G 채널
                multiArray[[0, 2, y, x] as [NSNumber]] = NSNumber(value: b) // B 채널
            }
        }
        
        return multiArray
    }
}

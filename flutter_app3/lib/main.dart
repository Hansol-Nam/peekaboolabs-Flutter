import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;
import 'package:permission_handler/permission_handler.dart';
import 'dart:ui' as ui;

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // 사용 가능한 카메라 목록 가져오기
  final cameras = await availableCameras();
  final firstCamera = cameras.first;

  runApp(MyApp(camera: firstCamera));
}

class MyApp extends StatelessWidget {
  final CameraDescription camera;

  const MyApp({super.key, required this.camera});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Face Emotion Recognition',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: FaceDetectionPage(camera: camera),
    );
  }
}

class FaceDetectionPage extends StatefulWidget {
  final CameraDescription camera;

  const FaceDetectionPage({super.key, required this.camera});

  @override
  _FaceDetectionPageState createState() => _FaceDetectionPageState();
}

class _FaceDetectionPageState extends State<FaceDetectionPage>
    with WidgetsBindingObserver {
  late CameraController _controller;
  late Future<void> _initializeControllerFuture;
  final FaceDetector _faceDetector = FaceDetector(
    options: FaceDetectorOptions(
      enableContours: true,
      enableClassification: true,
    ),
  );
  bool _isDetecting = false;
  List<Face> _faces = [];
  String _emotion = "알 수 없음";

  // MethodChannel 정의
  static const platform =
      MethodChannel('com.peekaboolabs.flutter_app3/emotion');

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _requestPermission();
    _controller = CameraController(
      widget.camera,
      ResolutionPreset.medium,
      enableAudio: false,
    );
    _initializeControllerFuture = _controller.initialize().then((_) {
      if (!mounted) return;
      setState(() {});
      _controller.startImageStream(_processCameraImage);
    });
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _controller.dispose();
    _faceDetector.close();
    super.dispose();
  }

  // 권한 요청
  Future<void> _requestPermission() async {
    var status = await Permission.camera.status;
    if (!status.isGranted) {
      await Permission.camera.request();
    }
  }

  // 카메라 프레임 처리
  void _processCameraImage(CameraImage image) async {
    if (_isDetecting) return;
    _isDetecting = true;

    try {
      final Size imageSize =
          Size(image.width.toDouble(), image.height.toDouble());

      // 카메라의 센서 방향을 기반으로 InputImageRotation 설정
      final imageRotation = _convertSensorOrientationToInputImageRotation(
          widget.camera.sensorOrientation);

      final inputImageFormat =
          InputImageFormatValue.fromRawValue(image.format.raw) ??
              InputImageFormat.nv21;

      final planeData = image.planes.map(
        (Plane plane) {
          return InputImagePlaneMetadata(
            bytesPerRow: plane.bytesPerRow,
            height: plane.height,
            width: plane.width,
          );
        },
      ).toList();

      final inputImageData = InputImageData(
        size: imageSize,
        imageRotation: imageRotation,
        inputImageFormat: inputImageFormat,
        planeData: planeData,
      );

      final inputImage = InputImage.fromBytes(
          bytes: _concatenatePlanes(image.planes),
          inputImageData: inputImageData);

      final faces = await _faceDetector.processImage(inputImage);

      if (mounted) {
        setState(() {
          _faces = faces;
        });
      }

      for (var face in faces) {
        // 얼굴의 boundingBox 추출
        final rect = face.boundingBox;

        // 카메라 프레임에서 얼굴 영역 추출
        Uint8List? faceBytes = await _extractFaceBytes(image, rect);

        if (faceBytes != null) {
          // 네이티브 코드로 얼굴 이미지 전송 및 감정 예측
          String emotion = await _getEmotionFromNative(faceBytes);
          setState(() {
            _emotion = emotion;
          });
        }
      }

      // 디버깅: 감지된 얼굴 수와 바운딩 박스 정보 출력
      print("Detected ${faces.length} face(s).");
      for (var face in faces) {
        print("Face boundingBox: ${face.boundingBox}");
      }
    } catch (e) {
      print("Error processing image: $e");
    } finally {
      _isDetecting = false;
    }
  }

  // 이미지의 모든 평면을 하나의 바이트 배열로 결합
  Uint8List _concatenatePlanes(List<Plane> planes) {
    final WriteBuffer allBytes = WriteBuffer();
    for (Plane plane in planes) {
      allBytes.putUint8List(plane.bytes);
    }
    return allBytes.done().buffer.asUint8List();
  }

  Future<Uint8List?> _extractFaceBytes(
      CameraImage image, Rect boundingBox) async {
    try {
      // CameraImage를 JPEG으로 변환하기 위해 YUV420을 RGB로 변환
      final img.Image convertedImage = _convertYUV420ToImage(image);

      // 얼굴 영역 추출
      final int imgWidth = convertedImage.width;
      final int imgHeight = convertedImage.height;

      // boundingBox의 비율을 실제 이미지 크기로 변환
      final double scaleX = imgWidth / image.width;
      final double scaleY = imgHeight / image.height;

      final int left =
          (boundingBox.left * scaleX).clamp(0, imgWidth - 1).toInt();
      final int top =
          (boundingBox.top * scaleY).clamp(0, imgHeight - 1).toInt();
      final int width =
          (boundingBox.width * scaleX).clamp(0, imgWidth - left).toInt();
      final int height =
          (boundingBox.height * scaleY).clamp(0, imgHeight - top).toInt();

      // 얼굴 영역을 자르기
      final img.Image faceImage =
          img.copyCrop(convertedImage, left, top, width, height);

      // 얼굴 이미지를 JPEG으로 인코딩
      final Uint8List faceBytes =
          Uint8List.fromList(img.encodeJpg(faceImage, quality: 80));

      return faceBytes;
    } catch (e) {
      print("Error extracting face bytes: $e");
      return null;
    }
  }

// YUV420을 RGB로 변환하는 메서드
  img.Image _convertYUV420ToImage(CameraImage image) {
    final int width = image.width;
    final int height = image.height;
    final int uvRowStride = image.planes[1].bytesPerRow;
    final int uvPixelStride = image.planes[1].bytesPerPixel!;

    final img.Image imgImage = img.Image(width, height); // Create Image buffer

    for (int y = 0; y < height; y++) {
      final int uvRow = y ~/ 2;
      for (int x = 0; x < width; x++) {
        final int uvCol = x ~/ 2;

        final int indexY = y * image.planes[0].bytesPerRow + x;
        final int indexU = uvRow * uvRowStride + uvCol * uvPixelStride;
        final int indexV = uvRow * uvRowStride + uvCol * uvPixelStride;

        final int yValue = image.planes[0].bytes[indexY];
        final int uValue = image.planes[1].bytes[indexU];
        final int vValue = image.planes[2].bytes[indexV];

        // YUV to RGB 변환
        double yDouble = yValue.toDouble();
        double uDouble = uValue.toDouble() - 128.0;
        double vDouble = vValue.toDouble() - 128.0;

        int r = (yDouble + 1.370705 * vDouble).clamp(0, 255).toInt();
        int g = (yDouble - 0.337633 * uDouble - 0.698001 * vDouble)
            .clamp(0, 255)
            .toInt();
        int b = (yDouble + 1.732446 * uDouble).clamp(0, 255).toInt();

        imgImage.setPixelRgba(x, y, r, g, b);
      }
    }

    return imgImage;
  }

  // 네이티브 코드와 통신하여 감정 예측
  Future<String> _getEmotionFromNative(Uint8List faceBytes) async {
    try {
      final String emotion = await platform.invokeMethod('getEmotion', {
        'faceBytes': faceBytes,
      });
      return emotion;
    } on PlatformException catch (e) {
      print("Failed to get emotion: '${e.message}'.");
      return "오류 발생";
    }
  }

  // 센서 방향을 ML Kit의 InputImageRotation으로 변환
  InputImageRotation _convertSensorOrientationToInputImageRotation(
      int sensorOrientation) {
    switch (sensorOrientation) {
      case 0:
        return InputImageRotation.rotation0deg;
      case 90:
        return InputImageRotation.rotation90deg;
      case 180:
        return InputImageRotation.rotation180deg;
      case 270:
        return InputImageRotation.rotation270deg;
      default:
        return InputImageRotation.rotation0deg;
    }
  }

  @override
  Widget build(BuildContext context) {
    if (!_controller.value.isInitialized) {
      return Scaffold(
        appBar: AppBar(title: const Text('Face Emotion Recognition')),
        body: const Center(child: CircularProgressIndicator()),
      );
    }
    return Scaffold(
      appBar: AppBar(title: const Text('Face Emotion Recognition')),
      body: Stack(
        fit: StackFit.expand,
        children: [
          AspectRatio(
            aspectRatio: _controller.value.aspectRatio,
            child: CameraPreview(_controller),
          ),
          // 바운딩 박스 그리기
          CustomPaint(
            painter: FacePainter(
              faces: _faces,
              imageSize: _controller.value.previewSize!,
              imageRotation: _convertSensorOrientationToInputImageRotation(
                  widget.camera.sensorOrientation),
              isFrontCamera:
                  widget.camera.lensDirection == CameraLensDirection.front,
            ),
          ),
          // 감정 결과 표시
          Positioned(
            bottom: 50,
            left: 0,
            right: 0,
            child: Center(
              child: Container(
                padding: EdgeInsets.all(10),
                color: Colors.black54,
                child: Text(
                  'Emotion: $_emotion',
                  style: TextStyle(color: Colors.white, fontSize: 20),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class FacePainter extends CustomPainter {
  final List<Face> faces;
  final Size imageSize;
  final InputImageRotation imageRotation;
  final bool isFrontCamera;

  FacePainter({
    required this.faces,
    required this.imageSize,
    required this.imageRotation,
    required this.isFrontCamera,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.redAccent
      ..strokeWidth = 2.0
      ..style = PaintingStyle.stroke;

    // 이미지의 회전에 따라 스케일링을 조정
    double scaleX, scaleY;

    switch (imageRotation) {
      case InputImageRotation.rotation90deg:
      case InputImageRotation.rotation270deg:
        // 이미지가 가로로 회전된 경우
        scaleX = size.width / imageSize.height;
        scaleY = size.height / imageSize.width;
        break;
      default:
        // 이미지가 세로로 회전된 경우
        scaleX = size.width / imageSize.width;
        scaleY = size.height / imageSize.height;
    }

    for (var face in faces) {
      final rect = face.boundingBox;

      // 이미지 회전에 따라 좌표를 조정
      Rect transformedRect;
      switch (imageRotation) {
        case InputImageRotation.rotation90deg:
          transformedRect = Rect.fromLTRB(
            rect.top * scaleX,
            rect.left * scaleY,
            rect.bottom * scaleX,
            rect.right * scaleY,
          );
          break;
        case InputImageRotation.rotation270deg:
          transformedRect = Rect.fromLTRB(
            size.width - rect.bottom * scaleX,
            rect.left * scaleY,
            size.width - rect.top * scaleX,
            rect.right * scaleY,
          );
          break;
        case InputImageRotation.rotation180deg:
          transformedRect = Rect.fromLTRB(
            size.width - rect.right * scaleX,
            size.height - rect.bottom * scaleY,
            size.width - rect.left * scaleX,
            size.height - rect.top * scaleY,
          );
          break;
        default:
          transformedRect = Rect.fromLTRB(
            rect.left * scaleX,
            rect.top * scaleY,
            rect.right * scaleX,
            rect.bottom * scaleY,
          );
      }

      // 프론트 카메라일 경우 좌우 반전
      if (isFrontCamera) {
        transformedRect = Rect.fromLTRB(
          size.width - transformedRect.right,
          transformedRect.top,
          size.width - transformedRect.left,
          transformedRect.bottom,
        );
      }

      // 디버깅: 변환된 바운딩 박스 좌표 출력
      print("Transformed Rect: $transformedRect");

      // 바운딩 박스 그리기
      canvas.drawRect(transformedRect, paint);
    }
  }

  @override
  bool shouldRepaint(FacePainter oldDelegate) {
    return oldDelegate.faces != faces ||
        oldDelegate.imageSize != imageSize ||
        oldDelegate.imageRotation != imageRotation ||
        oldDelegate.isFrontCamera != isFrontCamera;
  }
}

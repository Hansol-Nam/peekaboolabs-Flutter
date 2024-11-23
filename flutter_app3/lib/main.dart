import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'dart:typed_data';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:flutter/services.dart'; // DeviceOrientation을 위해 필요

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
      title: 'Face Detection',
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

  // 앱의 방향을 세로 모드로 고정했으므로, 항상 portraitUp으로 설정
  late DeviceOrientation _currentOrientation;

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

    // 앱이 세로 모드로 고정되었으므로, orientation을 portraitUp으로 설정
    _currentOrientation = DeviceOrientation.portraitUp;
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _controller.dispose();
    _faceDetector.close();
    super.dispose();
  }

  // 앱의 방향이 변경될 때 호출되지만, 세로 모드로 고정했으므로 추가 처리 불필요
  @override
  void didChangeMetrics() {
    // Orientation 고정으로 인해 아무 동작도 하지 않음
  }

  Future<void> _requestPermission() async {
    var status = await Permission.camera.status;
    if (!status.isGranted) {
      await Permission.camera.request();
    }
  }

  void _processCameraImage(CameraImage image) async {
    if (_isDetecting) return;
    _isDetecting = true;

    try {
      // WriteBuffer 없이 직접 바이트 결합
      List<int> bytes = [];
      for (var plane in image.planes) {
        bytes.addAll(plane.bytes);
      }
      Uint8List bytesUint8 = Uint8List.fromList(bytes);

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
          bytes: bytesUint8, inputImageData: inputImageData);

      final faces = await _faceDetector.processImage(inputImage);

      if (mounted) {
        setState(() {
          _faces = faces;
        });
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

  // 카메라 센서 방향(0, 90, 180, 270)을 ML Kit의 InputImageRotation으로 변환
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

  Widget _buildResults() {
    if (_faces.isEmpty) return Container();
    return CustomPaint(
      painter: FacePainter(
        faces: _faces,
        imageSize: _controller.value.previewSize!,
        imageRotation: _convertSensorOrientationToInputImageRotation(
            widget.camera.sensorOrientation),
        isFrontCamera: widget.camera.lensDirection == CameraLensDirection.front,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    if (!_controller.value.isInitialized) {
      return Scaffold(
        appBar: AppBar(title: const Text('Face Detection')),
        body: const Center(child: CircularProgressIndicator()),
      );
    }
    return Scaffold(
      appBar: AppBar(title: const Text('Face Detection')),
      body: Stack(
        fit: StackFit.expand,
        children: [
          AspectRatio(
            aspectRatio: _controller.value.aspectRatio,
            child: CameraPreview(_controller),
          ),
          _buildResults(),
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

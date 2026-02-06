import 'dart:async';
import 'dart:math';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_vision/flutter_vision.dart';
import 'package:isar/isar.dart';
import 'package:path_provider/path_provider.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:permission_handler/permission_handler.dart';
import 'package:printing/printing.dart';

part 'main.g.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setPreferredOrientations([
    DeviceOrientation.landscapeLeft,
    DeviceOrientation.landscapeRight,
  ]).then((_) => runApp(const PeopleCounterApp()));
}

class PeopleCounterApp extends StatelessWidget {
  const PeopleCounterApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Contador de Pessoas',
      theme: ThemeData.dark(useMaterial3: true),
      home: const CounterPage(),
    );
  }
}

class CounterPage extends StatefulWidget {
  const CounterPage({super.key});

  @override
  State<CounterPage> createState() => _CounterPageState();
}

class _CounterPageState extends State<CounterPage> {
  final FlutterVision _vision = FlutterVision();
  CameraController? _cameraController;
  Isar? _isar;

  bool _initializing = true;
  bool _cameraReady = false;
  bool _modelReady = false;
  bool _isDetecting = false;

  int _entraram = 0;
  int _sairam = 0;
  int _nextTrackId = 0;

  int _imageWidth = 0;
  int _imageHeight = 0;
  // Sensor orientation reported by the camera (degrees)
  int _sensorOrientation = 0;

  final Map<int, double> trackHistory = <int, double>{};
  final Map<int, Offset> _trackCenters = <int, Offset>{};

  List<DetectionBox> _boxes = <DetectionBox>[];

  @override
  void initState() {
    super.initState();
    _initialize();
  }

  Future<void> _initialize() async {
    try {
      await _requestPermissions();
      await _openIsar();
      await _loadModel();
      await _startCamera();
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Erro na inicialização: $e')),
        );
      }
    } finally {
      if (mounted) {
        setState(() {
          _initializing = false;
        });
      }
    }
  }

  Future<void> _requestPermissions() async {
    final cameraStatus = await Permission.camera.request();
    if (!cameraStatus.isGranted) {
      throw Exception('Permissão da câmera negada.');
    }
  }

  Future<void> _openIsar() async {
    final dir = await getApplicationDocumentsDirectory();
    _isar = await Isar.open(
      <CollectionSchema<dynamic>>[CountingLogSchema],
      directory: dir.path,
      name: 'people_counter_db',
    );
  }

  Future<void> _loadModel() async {
    await _vision.loadYoloModel(
      modelPath: 'assets/yolov8n.tflite',
      labels: 'assets/labels.txt',
      modelVersion: 'yolov8',
      quantization: false,
      numThreads: 4,
      useGpu: false,
      isAsset: true,
    );

    _modelReady = true;
  }

  Future<void> _startCamera() async {
    final cameras = await availableCameras();
    if (cameras.isEmpty) {
      throw Exception('Nenhuma câmera disponível.');
    }

    final selected = cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.back,
      orElse: () => cameras.first,
    );

    final controller = CameraController(
      selected,
      ResolutionPreset.medium,
      imageFormatGroup: ImageFormatGroup.yuv420,
      enableAudio: false,
    );

    await controller.initialize();
    // sensorOrientation: degrees the sensor is rotated relative to device
    _sensorOrientation = controller.description.sensorOrientation;
    _cameraController = controller;
    _cameraReady = true;

    await controller.startImageStream(_onCameraImage);
  }

  Future<void> _onCameraImage(CameraImage image) async {
    if (!_modelReady || _isDetecting || !mounted) return;

    _isDetecting = true;

    try {
      _imageWidth = image.width;
      _imageHeight = image.height;

      // Calculate rotation between sensor and device orientation so we can
      // correctly orient the image before running inference.
      final deviceOrientation = _cameraController?.value.deviceOrientation;
      final deviceDegrees = _deviceOrientationToDegrees(deviceOrientation);
      final rotationDegrees = (_sensorOrientation - deviceDegrees) % 360;
      if (rotationDegrees != 0) {
        // NOTE: Rotating YUV planes is non-trivial. If your inference library
        // accepts a rotation parameter, pass `rotationDegrees` to it here.
        // Otherwise implement rotation of the YUV420 planes before sending
        // to `_vision.yoloOnFrame`.
        // For now we log it and continue sending the raw frame.
        // TODO: add actual rotation handling if model requires it.
        // debugPrint('Frame rotation needed: $rotationDegrees');
      }

      final bytesList = image.planes.map((p) => p.bytes).toList();

      final rawDetections = await _vision.yoloOnFrame(
        bytesList: bytesList,
        imageHeight: image.height,
        imageWidth: image.width,
        iouThreshold: 0.4,
        confThreshold: 0.4,
        classThreshold: 0.5,
      );

      final detections = _extractPeopleDetections(rawDetections);
      _processTrackingAndCounting(detections);

      if (mounted) {
        setState(() {
          _boxes = detections;
        });
      }
    } catch (_) {
      // Mantém o stream ativo mesmo se houver erro em um frame.
    } finally {
      _isDetecting = false;
    }
  }

  List<DetectionBox> _extractPeopleDetections(
    List<Map<String, dynamic>> rawDetections,
  ) {
    final boxes = <DetectionBox>[];

    for (final item in rawDetections) {
      final tag = (item['tag'] ?? '').toString().toLowerCase();
      if (!tag.contains('person')) continue;

      final dynamic rawBox = item['box'];
      if (rawBox is! List || rawBox.length < 4) continue;

      final x1 = _toDouble(rawBox[0]);
      final y1 = _toDouble(rawBox[1]);
      final x2 = _toDouble(rawBox[2]);
      final y2 = _toDouble(rawBox[3]);
      final conf = rawBox.length > 4 ? _toDouble(rawBox[4]) : 0.0;

      if (_imageWidth == 0 || _imageHeight == 0) continue;

      final left = x1.clamp(0.0, _imageWidth.toDouble());
      final top = y1.clamp(0.0, _imageHeight.toDouble());
      final right = x2.clamp(0.0, _imageWidth.toDouble());
      final bottom = y2.clamp(0.0, _imageHeight.toDouble());

      if (right <= left || bottom <= top) continue;

      boxes.add(
        DetectionBox(
          rect: Rect.fromLTRB(left, top, right, bottom),
          tag: tag,
          confidence: conf,
        ),
      );
    }

    return boxes;
  }

  void _processTrackingAndCounting(List<DetectionBox> detections) {
    final newTrackHistory = <int, double>{};
    final newCenters = <int, Offset>{};
    final usedPreviousIds = <int>{};

    for (final d in detections) {
      final center = d.rect.center;
      final centerNormX = (center.dx / _imageWidth).clamp(0.0, 1.0);
      final centerNormY = (center.dy / _imageHeight).clamp(0.0, 1.0);

      int? assignedId = _findNearestTrack(
        currentCenter: Offset(centerNormX, centerNormY),
        usedIds: usedPreviousIds,
      );

      assignedId ??= ++_nextTrackId;
      d.trackId = assignedId;

      final previousY = trackHistory[assignedId];
      if (previousY != null) {
        if (previousY < 0.5 && centerNormY >= 0.5) {
          _entraram++;
          _saveLog('entrada');
        } else if (previousY > 0.5 && centerNormY <= 0.5) {
          _sairam++;
          _saveLog('saida');
        }
      }

      newTrackHistory[assignedId] = centerNormY;
      newCenters[assignedId] = Offset(centerNormX, centerNormY);
      usedPreviousIds.add(assignedId);
    }

    trackHistory
      ..clear()
      ..addAll(newTrackHistory);

    _trackCenters
      ..clear()
      ..addAll(newCenters);
  }

  int? _findNearestTrack({
    required Offset currentCenter,
    required Set<int> usedIds,
  }) {
    int? nearestId;
    double nearestDistance = double.infinity;

    for (final entry in _trackCenters.entries) {
      final id = entry.key;
      if (usedIds.contains(id)) continue;

      final prev = entry.value;
      final distance = sqrt(
        pow(currentCenter.dx - prev.dx, 2) +
            pow(currentCenter.dy - prev.dy, 2),
      );

      if (distance < nearestDistance) {
        nearestDistance = distance;
        nearestId = id;
      }
    }

    if (nearestDistance > 0.12) {
      return null;
    }

    return nearestId;
  }

  void _saveLog(String type) {
    final isar = _isar;
    if (isar == null) return;

    final log = CountingLog(
      timestamp: DateTime.now(),
      type: type,
    );

    unawaited(
      isar.writeTxn(() async {
        await isar.collection<CountingLog>().put(log);
      }),
    );
  }

  int _deviceOrientationToDegrees(DeviceOrientation? orientation) {
    switch (orientation) {
      case DeviceOrientation.portraitUp:
        return 0;
      case DeviceOrientation.landscapeLeft:
        return 90;
      case DeviceOrientation.portraitDown:
        return 180;
      case DeviceOrientation.landscapeRight:
        return 270;
      default:
        return 0;
    }
  }

  Future<void> _exportPdf() async {
    final isar = _isar;
    if (isar == null) return;

    final logs = await isar.collection<CountingLog>().where().findAll();
    logs.sort((a, b) => a.timestamp.compareTo(b.timestamp));

    final doc = pw.Document();
    final rows = logs
        .map(
          (l) => <String>[
            _formatDateTime(l.timestamp),
            l.type,
          ],
        )
        .toList();

    doc.addPage(
      pw.MultiPage(
        build: (context) => [
          pw.Header(
            level: 0,
            child: pw.Text('Relatório de Contagem de Pessoas'),
          ),
          pw.SizedBox(height: 8),
          pw.Table.fromTextArray(
            headers: const <String>['Data/Hora', 'Tipo'],
            data: rows,
          ),
        ],
      ),
    );

    await Printing.layoutPdf(onLayout: (format) async => doc.save());
  }

  String _formatDateTime(DateTime dt) {
    String two(int n) => n.toString().padLeft(2, '0');
    return '${two(dt.day)}/${two(dt.month)}/${dt.year} '
        '${two(dt.hour)}:${two(dt.minute)}:${two(dt.second)}';
  }

  double _toDouble(dynamic value) {
    if (value is double) return value;
    if (value is int) return value.toDouble();
    if (value is String) return double.tryParse(value) ?? 0.0;
    return 0.0;
  }

  @override
  void dispose() {
    final controller = _cameraController;
    if (controller != null) {
      if (controller.value.isStreamingImages) {
        unawaited(controller.stopImageStream());
      }
      controller.dispose();
    }

    unawaited(_vision.closeYoloModel());
    final isar = _isar;
    if (isar != null) {
      unawaited(isar.close().then((_) {}));
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Contabilizador de pessoas MJ'),
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _exportPdf,
        icon: const Icon(Icons.picture_as_pdf),
        label: const Text('Exportar PDF'),
      ),
      body: _buildBody(),
    );
  }

  Widget _buildBody() {
    if (_initializing) {
      return const Center(child: CircularProgressIndicator());
    }

    if (!_cameraReady || _cameraController == null) {
      return const Center(child: Text('Câmera não inicializada.'));
    }

    final controller = _cameraController!;

    return Stack(
      children: [
        Center(
          child: AspectRatio(
            aspectRatio: controller.value.aspectRatio,
            child: Stack(
              fit: StackFit.expand,
              children: [
                CameraPreview(controller),
                CustomPaint(
                  painter: DetectionPainter(
                    detections: _boxes,
                    imageSize: Size(
                      _imageWidth.toDouble(),
                      _imageHeight.toDouble(),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
        Positioned(
          top: 16,
          left: 16,
          right: 16,
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              _CounterChip(label: 'Entraram', value: _entraram),
              _CounterChip(label: 'Saíram', value: _sairam),
            ],
          ),
        ),
      ],
    );
  }
}

class _CounterChip extends StatelessWidget {
  const _CounterChip({required this.label, required this.value});

  final String label;
  final int value;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.65),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Column(
        children: [
          Text(label, style: const TextStyle(fontSize: 13)),
          const SizedBox(height: 4),
          Text(
            '$value',
            style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
          ),
        ],
      ),
    );
  }
}

class DetectionPainter extends CustomPainter {
  DetectionPainter({
    required this.detections,
    required this.imageSize,
  });

  final List<DetectionBox> detections;
  final Size imageSize;

  @override
  void paint(Canvas canvas, Size size) {
    final linePaint = Paint()
      ..color = Colors.red
      ..strokeWidth = 2;

    canvas.drawLine(
      Offset(0, size.height * 0.5),
      Offset(size.width, size.height * 0.5),
      linePaint,
    );

    if (imageSize.width <= 0 || imageSize.height <= 0) {
      return;
    }

    final scaleX = size.width / imageSize.width;
    final scaleY = size.height / imageSize.height;

    final boxPaint = Paint()
      ..color = Colors.green
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;

    for (final d in detections) {
      final rect = Rect.fromLTRB(
        d.rect.left * scaleX,
        d.rect.top * scaleY,
        d.rect.right * scaleX,
        d.rect.bottom * scaleY,
      );

      canvas.drawRect(rect, boxPaint);

      final text = 'ID ${d.trackId ?? '-'} ${d.tag} '
          '${(d.confidence * 100).toStringAsFixed(0)}%';

      final painter = TextPainter(
        text: TextSpan(
          text: text,
          style: const TextStyle(
            color: Colors.greenAccent,
            fontSize: 12,
            fontWeight: FontWeight.w600,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout(maxWidth: size.width);

      final offset = Offset(
        rect.left,
        max(0, rect.top - 16),
      );

      painter.paint(canvas, offset);
    }
  }

  @override
  bool shouldRepaint(covariant DetectionPainter oldDelegate) {
    return oldDelegate.detections != detections ||
        oldDelegate.imageSize != imageSize;
  }
}

class DetectionBox {
  DetectionBox({
    required this.rect,
    required this.tag,
    required this.confidence,
    this.trackId,
  });

  final Rect rect;
  final String tag;
  final double confidence;
  int? trackId;
}

@collection
class CountingLog {
  CountingLog({
    this.id = Isar.autoIncrement,
    required this.timestamp,
    required this.type,
  });

  Id id;
  DateTime timestamp;
  String type;
}

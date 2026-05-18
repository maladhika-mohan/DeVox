import 'dart:io';
import 'package:record/record.dart';
import 'package:path_provider/path_provider.dart';

class AudioRecorderService {
  final AudioRecorder _recorder = AudioRecorder();
  String? _currentPath;

  bool _isRecording = false;
  bool get isRecording => _isRecording;

  /// Start recording audio
  Future<bool> startRecording() async {
    if (await _recorder.hasPermission()) {
      final dir = await getTemporaryDirectory();
      _currentPath = '${dir.path}/saamai_recording.wav';

      await _recorder.start(
        const RecordConfig(
          encoder: AudioEncoder.wav,
          sampleRate: 44100,
          numChannels: 1,
          bitRate: 128000,
        ),
        path: _currentPath!,
      );

      _isRecording = true;
      return true;
    }
    return false;
  }

  /// Stop recording and return the file path
  Future<String?> stopRecording() async {
    if (_isRecording) {
      final path = await _recorder.stop();
      _isRecording = false;
      return path ?? _currentPath;
    }
    return null;
  }

  /// Get recording duration stream
  Stream<RecordState> get onStateChanged => _recorder.onStateChanged();

  void dispose() {
    _recorder.dispose();
  }
}

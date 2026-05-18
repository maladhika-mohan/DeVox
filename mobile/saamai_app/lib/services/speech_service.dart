import 'package:speech_to_text/speech_to_text.dart';

class SpeechService {
  final SpeechToText _speech = SpeechToText();
  bool _isInitialized = false;

  /// Initialize speech recognition
  Future<bool> initialize() async {
    _isInitialized = await _speech.initialize();
    return _isInitialized;
  }

  /// Listen for voice command and extract instrument name
  Future<String?> listenForInstrument() async {
    if (!_isInitialized) {
      final ok = await initialize();
      if (!ok) return null;
    }

    String? result;

    await _speech.listen(
      onResult: (speechResult) {
        if (speechResult.finalResult) {
          result = _extractInstrument(speechResult.recognizedWords);
        }
      },
      listenFor: const Duration(seconds: 5),
      pauseFor: const Duration(seconds: 2),
    );

    // Wait for result
    await Future.delayed(const Duration(seconds: 6));
    await _speech.stop();

    return result;
  }

  /// Extract instrument name from spoken text
  String? _extractInstrument(String text) {
    final lower = text.toLowerCase();

    // Map of keywords to instrument names
    const instrumentKeywords = {
      'piano': 'Piano',
      'violin': 'Violin',
      'viola': 'Viola',
      'cello': 'Cello',
      'guitar': 'Guitar (Nylon)',
      'acoustic guitar': 'Guitar (Steel)',
      'electric guitar': 'Electric Guitar (Clean)',
      'flute': 'Flute',
      'clarinet': 'Clarinet',
      'saxophone': 'Saxophone (Alto)',
      'sax': 'Saxophone (Alto)',
      'trumpet': 'Trumpet',
      'trombone': 'Trombone',
      'french horn': 'French Horn',
      'harp': 'Harp',
      'sitar': 'Sitar',
      'veena': 'Veena',
      'vina': 'Veena',
      'tabla': 'Tabla',
      'organ': 'Organ',
      'harmonica': 'Harmonica',
      'oboe': 'Oboe',
      'bassoon': 'Bassoon',
      'banjo': 'Banjo',
      'xylophone': 'Xylophone',
      'marimba': 'Marimba',
      'kalimba': 'Kalimba',
      'bagpipe': 'Bagpipe',
      'accordion': 'Accordion',
    };

    for (final entry in instrumentKeywords.entries) {
      if (lower.contains(entry.key)) {
        return entry.value;
      }
    }

    return null;
  }

  bool get isListening => _speech.isListening;

  void dispose() {
    _speech.stop();
  }
}

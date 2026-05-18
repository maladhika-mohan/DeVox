import 'dart:io';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:path_provider/path_provider.dart';

class ApiService {
  // Change this to your server IP/URL
  static const String baseUrl = 'http://10.0.2.2:8000'; // Android emulator → localhost
  // For physical device, use your machine's local IP:
  // static const String baseUrl = 'http://192.168.1.X:8000';

  /// Fetch available instruments from the API
  static Future<List<String>> getInstruments() async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/instruments'));
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return List<String>.from(data['instruments']);
      }
    } catch (e) {
      print('Error fetching instruments: $e');
    }
    // Fallback list if API is unreachable
    return [
      'Piano', 'Violin', 'Flute', 'Guitar (Nylon)', 'Guitar (Steel)',
      'Saxophone (Alto)', 'Cello', 'Trumpet', 'Clarinet', 'Harp',
      'Sitar', 'Veena', 'Electric Piano', 'Organ', 'Harmonica',
    ];
  }

  /// Send audio file + instrument choice, receive synthesized WAV
  static Future<File?> convertAudio({
    required String audioPath,
    required String instrument,
  }) async {
    try {
      final uri = Uri.parse('$baseUrl/convert');
      final request = http.MultipartRequest('POST', uri);

      request.files.add(
        await http.MultipartFile.fromPath('audio', audioPath),
      );
      request.fields['instrument'] = instrument;

      final streamedResponse = await request.send().timeout(
        const Duration(seconds: 120),
      );

      if (streamedResponse.statusCode == 200) {
        final bytes = await streamedResponse.stream.toBytes();
        final dir = await getTemporaryDirectory();
        final safeName = instrument.replaceAll(' ', '_').replaceAll('(', '').replaceAll(')', '');
        final outputFile = File('${dir.path}/saamai_$safeName.wav');
        await outputFile.writeAsBytes(bytes);
        return outputFile;
      } else {
        final body = await streamedResponse.stream.bytesToString();
        print('API error ${streamedResponse.statusCode}: $body');
        return null;
      }
    } catch (e) {
      print('Convert error: $e');
      return null;
    }
  }
}

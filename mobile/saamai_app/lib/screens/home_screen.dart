import 'dart:io';
import 'package:flutter/material.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:permission_handler/permission_handler.dart';
import '../services/api_service.dart';
import '../services/audio_recorder.dart';
import '../services/speech_service.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with TickerProviderStateMixin {
  final AudioRecorderService _recorder = AudioRecorderService();
  final AudioPlayer _player = AudioPlayer();
  final SpeechService _speech = SpeechService();

  List<String> _instruments = [];
  String _selectedInstrument = 'Piano';
  String? _recordedPath;
  File? _outputFile;

  bool _isRecording = false;
  bool _isProcessing = false;
  bool _isPlaying = false;
  String _statusMessage = 'Tap the mic to record your melody';

  late AnimationController _pulseController;

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1000),
    )..repeat(reverse: true);

    _loadInstruments();
    _requestPermissions();
    _speech.initialize();

    _player.onPlayerStateChanged.listen((state) {
      setState(() => _isPlaying = state == PlayerState.playing);
    });
  }

  Future<void> _requestPermissions() async {
    await [Permission.microphone, Permission.speech].request();
  }

  Future<void> _loadInstruments() async {
    final instruments = await ApiService.getInstruments();
    setState(() => _instruments = instruments);
  }

  Future<void> _toggleRecording() async {
    if (_isRecording) {
      final path = await _recorder.stopRecording();
      setState(() {
        _isRecording = false;
        _recordedPath = path;
        _statusMessage = 'Recording saved! Choose instrument & convert.';
      });
    } else {
      final started = await _recorder.startRecording();
      if (started) {
        setState(() {
          _isRecording = true;
          _outputFile = null;
          _statusMessage = 'Listening... sing or hum your melody';
        });
      } else {
        setState(() => _statusMessage = 'Microphone permission denied');
      }
    }
  }

  Future<void> _convertAudio() async {
    if (_recordedPath == null) {
      setState(() => _statusMessage = 'Record something first!');
      return;
    }

    setState(() {
      _isProcessing = true;
      _statusMessage = 'Converting to $_selectedInstrument...';
    });

    final result = await ApiService.convertAudio(
      audioPath: _recordedPath!,
      instrument: _selectedInstrument,
    );

    setState(() {
      _isProcessing = false;
      if (result != null) {
        _outputFile = result;
        _statusMessage = '✓ $_selectedInstrument version ready!';
      } else {
        _statusMessage = 'Conversion failed. Check server connection.';
      }
    });
  }

  Future<void> _playOutput() async {
    if (_outputFile != null && _outputFile!.existsSync()) {
      if (_isPlaying) {
        await _player.stop();
      } else {
        await _player.play(DeviceFileSource(_outputFile!.path));
      }
    }
  }

  Future<void> _listenForVoiceCommand() async {
    setState(() => _statusMessage = 'Listening for instrument name...');

    final instrument = await _speech.listenForInstrument();

    if (instrument != null) {
      setState(() {
        _selectedInstrument = instrument;
        _statusMessage = 'Got it! Converting to $instrument...';
      });
      if (_recordedPath != null) {
        await _convertAudio();
      } else {
        setState(() => _statusMessage = 'Say the instrument, then record!');
      }
    } else {
      setState(() => _statusMessage = 'Didn\'t catch that. Try again.');
    }
  }

  @override
  void dispose() {
    _recorder.dispose();
    _player.dispose();
    _speech.dispose();
    _pulseController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Color(0xFF0F0A2A), Color(0xFF1A1145), Color(0xFF0D0820)],
          ),
        ),
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 24),
            child: Column(
              children: [
                const SizedBox(height: 40),
                _buildHeader(),
                const SizedBox(height: 50),
                _buildRecordButton(),
                const SizedBox(height: 20),
                _buildStatus(),
                const SizedBox(height: 40),
                _buildInstrumentSelector(),
                const SizedBox(height: 20),
                _buildActionButtons(),
                const SizedBox(height: 30),
                if (_outputFile != null) _buildPlayer(),
                const Spacer(),
                _buildBottomHint(),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Column(
      children: [
        ShaderMask(
          shaderCallback: (bounds) => const LinearGradient(
            colors: [Color(0xFFA78BFA), Color(0xFF60A5FA), Color(0xFF34D399)],
          ).createShader(bounds),
          child: const Text(
            'Saamai',
            style: TextStyle(
              fontSize: 42,
              fontWeight: FontWeight.w900,
              color: Colors.white,
            ),
          ),
        ),
        const SizedBox(height: 4),
        Text(
          'Hum it. Hear it. In any instrument.',
          style: TextStyle(fontSize: 14, color: Colors.white.withValues(alpha: 0.6)),
        ),
      ],
    );
  }

  Widget _buildRecordButton() {
    return GestureDetector(
      onTap: _isProcessing ? null : _toggleRecording,
      child: AnimatedBuilder(
        animation: _pulseController,
        builder: (context, child) {
          final scale = _isRecording ? 1.0 + (_pulseController.value * 0.08) : 1.0;
          return Transform.scale(
            scale: scale,
            child: Container(
              width: 140,
              height: 140,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                gradient: LinearGradient(
                  colors: _isRecording
                      ? [const Color(0xFFEF4444), const Color(0xFFDC2626)]
                      : [const Color(0xFF6366F1), const Color(0xFF8B5CF6)],
                ),
                boxShadow: [
                  BoxShadow(
                    color: (_isRecording ? const Color(0xFFEF4444) : const Color(0xFF6366F1))
                        .withValues(alpha: 0.4),
                    blurRadius: 30,
                    spreadRadius: 5,
                  ),
                ],
              ),
              child: Icon(
                _isRecording ? Icons.stop_rounded : Icons.mic_rounded,
                size: 56,
                color: Colors.white,
              ),
            ),
          );
        },
      ),
    );
  }

  Widget _buildStatus() {
    return Text(
      _statusMessage,
      textAlign: TextAlign.center,
      style: TextStyle(fontSize: 14, color: Colors.white.withValues(alpha: 0.7)),
    );
  }

  Widget _buildInstrumentSelector() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white.withValues(alpha: 0.1)),
      ),
      child: DropdownButtonHideUnderline(
        child: DropdownButton<String>(
          value: _instruments.contains(_selectedInstrument) ? _selectedInstrument : null,
          hint: const Text('Select Instrument', style: TextStyle(color: Colors.white54)),
          isExpanded: true,
          dropdownColor: const Color(0xFF1E1B4B),
          style: const TextStyle(color: Colors.white, fontSize: 16),
          items: _instruments.map((inst) {
            return DropdownMenuItem(value: inst, child: Text(inst));
          }).toList(),
          onChanged: (val) {
            if (val != null) setState(() => _selectedInstrument = val);
          },
        ),
      ),
    );
  }

  Widget _buildActionButtons() {
    return Row(
      children: [
        Expanded(
          child: GestureDetector(
            onTap: _listenForVoiceCommand,
            child: Container(
              padding: const EdgeInsets.symmetric(vertical: 16),
              decoration: BoxDecoration(
                color: const Color(0xFF8B5CF6).withValues(alpha: 0.15),
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: const Color(0xFF8B5CF6).withValues(alpha: 0.3)),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Icon(Icons.record_voice_over_rounded, color: Color(0xFF8B5CF6), size: 20),
                  const SizedBox(width: 8),
                  Text('Voice', style: TextStyle(color: Colors.white.withValues(alpha: 0.9), fontSize: 13, fontWeight: FontWeight.w600)),
                ],
              ),
            ),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          flex: 2,
          child: GestureDetector(
            onTap: _isProcessing ? null : _convertAudio,
            child: Container(
              padding: const EdgeInsets.symmetric(vertical: 16),
              decoration: BoxDecoration(
                color: const Color(0xFF6366F1).withValues(alpha: 0.15),
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: const Color(0xFF6366F1).withValues(alpha: 0.3)),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  if (_isProcessing)
                    const SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2, color: Color(0xFF6366F1)))
                  else
                    const Icon(Icons.music_note_rounded, color: Color(0xFF6366F1), size: 20),
                  const SizedBox(width: 8),
                  Flexible(
                    child: Text(
                      'Convert',
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(color: Colors.white.withValues(alpha: 0.9), fontSize: 13, fontWeight: FontWeight.w600),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildPlayer() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.06),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: const Color(0xFF34D399).withValues(alpha: 0.3)),
      ),
      child: Row(
        children: [
          GestureDetector(
            onTap: _playOutput,
            child: Container(
              width: 56,
              height: 56,
              decoration: const BoxDecoration(
                shape: BoxShape.circle,
                gradient: LinearGradient(colors: [Color(0xFF10B981), Color(0xFF34D399)]),
              ),
              child: Icon(
                _isPlaying ? Icons.pause_rounded : Icons.play_arrow_rounded,
                color: Colors.white,
                size: 32,
              ),
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  '$_selectedInstrument Version',
                  style: const TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.w600),
                ),
                const SizedBox(height: 4),
                Text(
                  'Tap to ${_isPlaying ? "pause" : "play"}',
                  style: TextStyle(color: Colors.white.withValues(alpha: 0.5), fontSize: 12),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildBottomHint() {
    return Padding(
      padding: const EdgeInsets.only(bottom: 20),
      child: Text(
        'Say "Hey Saamai, play this in violin"',
        style: TextStyle(fontSize: 12, color: Colors.white.withValues(alpha: 0.3), fontStyle: FontStyle.italic),
      ),
    );
  }
}

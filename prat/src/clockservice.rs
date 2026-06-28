//! prat::clockservice.rs
//!
//! Ableton Link-based basic clock service with dedicated timing thread
//!
//! Basic Operation:
//!
//! use prat::ClockService;
//!
//! 1. Use create a new ClockService instance using the builder:
//!    let clock = ClockService::with().tempo(120.0).quantum(4.0).build();
//!
//! 2. Start the Ableton Link session by spawning the timing thread:
//!    clock.start_thread().unwrap();
//!
//! 3. Start the clock:
//!    clock.start_clock().unwrap();
//!
//! 4. Subscribe to BeatEvent broadcast to receive timing events:
//!    let beat_rx = clock.subscribe_to_beats().unwrap();
//!
use crate::timing_thread::{TimingThread, TimingThreadCommand, TimingThreadState};

use rusty_link::AblLink;
use std::{
    sync::{atomic::AtomicBool, Arc, Mutex},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};
use tokio::sync::broadcast;

/// Callback function convenience type.
pub type BeatCallback = Box<dyn Fn(BeatEvent) + Send + 'static>;

/// The entry point for the Ableton Link-based clock service.
///
/// Starts a dedicated timing thread for high stability and accuracy.
/// The timing thread is responsible for:
/// - Maintaining the timing grid
/// - Sending beat events to the main thread
/// - Sending session state snapshots to the main thread
/// - Handling control commands from the main thread
pub struct ClockService {
    // Link components - shared between threads
    link: Arc<AblLink>,

    // Shared state
    // This is wrapped in an Arc for non-blocking, thread-safe access with
    // changes immediately reflected in the timing thread.
    session_params: Arc<Mutex<SessionParams>>, // fields extracted from the session state snapshot.
    start_stop_sync_enabled: Arc<AtomicBool>,
    clock_is_running: Arc<AtomicBool>,

    // Configuration
    ppqn: u8,
    is_sending_ticks: bool,
    beat_callback: Option<BeatCallback>,

    // Beat channels
    beat_tx: Option<broadcast::Sender<BeatEvent>>, // a channel to which clients can subscribe to receive beat events
    beat_rx: Option<broadcast::Receiver<BeatEvent>>,
    beat_buffer_size: usize,

    // Tick channels
    tick_tx: Option<broadcast::Sender<TickEvent>>, // a channel to which clients can subscribe to receive tick events
    tick_rx: Option<broadcast::Receiver<TickEvent>>,
    tick_buffer_size: usize,

    // Thread control
    timing_thread_join: Option<JoinHandle<()>>,
    control_tx: Option<broadcast::Sender<TimingThreadCommand>>,

    // Debug messages
    debug: bool,
}

/// Builder for ClockService.
pub struct ClockServiceBuilder {
    session_params: SessionParams,
    ppqn: u8,
    is_sending_ticks: bool,
    beat_buffer_size: usize,
    tick_buffer_size: usize,
    enable_start_stop_sync: bool,
    debug: bool,
}

/// BeatEvents are generated on every beat subdivision.
#[derive(Debug, Clone)]
pub struct BeatEvent {
    pub subdivisions: Vec<BeatSubdivision>, // the beat subdivisions that occurred
    pub subdivision_counts: Vec<u64>,       // the number of times each subdivision has occurred
    pub beat: f64,                          // Link Beat time
    pub phase: f64,                         // Link Beat phase
    pub secs_since_start: f64,              // the time since the clock started in seconds
}

/// TickEvents are generated on every tick of the clock.
#[derive(Debug, Copy, Clone)]
pub struct TickEvent {
    pub tick_count: u64,       // number of ticks since the clock started
    pub tick_time: i64,        // host time of this tick, microseconds
    pub tick_instant: Instant, // clock service's local instant at this tick
    pub next_tick_time: i64,   // host time of the next tick, microseconds
    pub secs_since_start: f64, // the time since the clock started in seconds
}

#[derive(Debug, Copy, Clone)]
pub struct SessionParams {
    pub tempo: f64,
    pub quantum: f64,
    pub link_is_playing: bool,
}

/// Offers an interface to send TimingCommands to the timing thread, without needing a mutable reference to the ClockService.
#[derive(Clone)]
pub struct TimingThreadControlChannel {
    control_tx: Option<broadcast::Sender<TimingThreadCommand>>,
}

impl ClockServiceBuilder {
    // All the defaults are set here.
    pub fn new() -> Self {
        Self {
            session_params: SessionParams {
                tempo: 120.0, // default tempo in bpm
                quantum: 4.0, // default quantum in beats
                link_is_playing: false,
            },
            ppqn: 24, // 24 is the default PPQN (same as MIDI)
            is_sending_ticks: false,
            enable_start_stop_sync: false,
            beat_buffer_size: 128,
            tick_buffer_size: 512,
            debug: false,
        }
    }

    /// Set the initial tempo in bpm. This will be overridden if joining an existing Link session.
    pub fn tempo(mut self, tempo: f64) -> Self {
        self.session_params.tempo = tempo;
        self
    }

    /// Set the number of beats per phase reset.
    pub fn quantum(mut self, quantum: f64) -> Self {
        self.session_params.quantum = quantum;
        self
    }

    /// Enable start/stop sync with other devices.
    /// - If enabled, the clock service will start the Link session in sync with other devices.
    /// - Default is false -- the clock service will be independent of other devices
    pub fn enable_start_stop_sync(mut self) -> Self {
        self.enable_start_stop_sync = true;
        self
    }

    /// Set the pulses per quarter note.
    pub fn ppqn(mut self, ppqn: u8) -> Self {
        self.ppqn = ppqn;
        self
    }

    /// Enable sending tick events to the main thread.
    /// - Default: false -- only beats are sent.
    pub fn enable_ticks(mut self) -> Self {
        self.is_sending_ticks = true;
        self
    }

    /// Set the buffer size for the beat channel.
    pub fn beat_buffer_size(mut self, size: usize) -> Self {
        self.beat_buffer_size = size;
        self
    }

    /// Set the buffer size for the tick channel.
    pub fn tick_buffer_size(mut self, size: usize) -> Self {
        self.tick_buffer_size = size;
        self
    }

    /// Enable debug messages in the console.
    pub fn debug(mut self) -> Self {
        self.debug = true;
        self
    }

    /// Build the ClockService.
    pub fn build(self) -> ClockService {
        let clock = ClockService {
            link: Arc::new(AblLink::new(self.session_params.tempo)),
            session_params: Arc::new(Mutex::new(self.session_params)),
            start_stop_sync_enabled: Arc::new(AtomicBool::new(self.enable_start_stop_sync)),
            clock_is_running: Arc::new(AtomicBool::new(false)),
            ppqn: self.ppqn,
            is_sending_ticks: self.is_sending_ticks,
            beat_callback: None,
            beat_tx: None,
            beat_rx: None,
            beat_buffer_size: self.beat_buffer_size,
            tick_tx: None,
            tick_rx: None,
            tick_buffer_size: self.tick_buffer_size,
            timing_thread_join: None,
            control_tx: None,
            debug: self.debug,
        };

        clock.link.enable(true);

        clock
            .link
            .enable_start_stop_sync(self.enable_start_stop_sync);

        clock
    }
}

impl Default for ClockServiceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ClockService {
    /// Builder function to create a new ClockService.
    ///
    /// ## Chainable methods:
    /// - .tempo(f64) -- set initial tempo in bpm. This will be overridden if joining an existing session. Default is 120.0
    /// - .quantum(f64) -- set the number of beats per phase reset. Each Link client can set its own quantum. Default is 4.0
    /// - .ppqn(u8) -- set the pulses per quarter note. Default is 24 (same as MIDI)
    /// - .enable_start_stop_sync() -- enable start/stop sync with other devices. Default is false
    /// - .enable_ticks() -- enable sending tick events to the main thread. Default is false
    /// - .beat_buffer_size(usize) -- set the buffer size for the beat channel. Default is 128
    /// - .tick_buffer_size(usize) -- set the buffer size for the tick channel. Default is 512
    /// - .debug() -- enable debug messages in the console
    /// - .build() -- build the ClockService
    pub fn with() -> ClockServiceBuilder {
        ClockServiceBuilder::new()
    }

    /// Create a new ClockService with the given `tempo` and `quantum`, and default values for the rest of the parameters.
    pub fn new(tempo: f64, quantum: f64) -> Self {
        Self::with().tempo(tempo).quantum(quantum).build()
    }

    /// Start the clock service on a new thread
    pub fn start_thread(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let Some(mut timing_thread) = self.init_timing_thread() else {
            return Err("Prat: Failed to create Timing Thread".into());
        };

        if self.debug {
            println!("Prat: Starting ClockService with {} ppqn...", self.ppqn);
        }
        if self.timing_thread_join.is_some() {
            return Err("Prat: Clock service already running".into());
        }

        let debug = self.debug;

        let timing_thread_handle = thread::Builder::new()
            .name("clock-timing".to_string())
            .spawn(move || {
                let _rt_handle =
                    audio_thread_priority::promote_current_thread_to_real_time(256, 48000)
                        .map_err(|e| eprintln!("Prat: real-time promotion failed: {e}"))
                        .ok();
                if debug {
                    println!("Prat: clock-timing thread started (RT: {})", _rt_handle.is_some());
                }
                timing_thread.run()
            })?;

        // Save thread handle for easy access
        self.timing_thread_join = Some(timing_thread_handle);

        // Wait a short time to ensure the thread is running
        thread::sleep(Duration::from_millis(100));

        Ok(())
    }

    /// Initialize and integrate a new TimingThread from this ClockService.
    pub fn init_timing_thread(&mut self) -> Option<TimingThread> {
        let (beat_tx, beat_rx) = broadcast::channel(self.beat_buffer_size);
        let (tick_tx, tick_rx) = broadcast::channel(self.tick_buffer_size);

        let (control_tx, control_rx) = broadcast::channel(32);

        let new_clock_thread_state = TimingThreadState::new(
            Arc::clone(&self.link),
            self.session_params.clone(),
            Arc::clone(&self.clock_is_running),
            Arc::clone(&self.start_stop_sync_enabled),
            self.ppqn,
            self.debug,
        );

        let timing_thread = TimingThread {
            ppqn: self.ppqn,
            is_sending_ticks: self.is_sending_ticks,
            debug: self.debug,
            state: new_clock_thread_state,
            beat_tx: beat_tx.clone(),
            tick_tx: tick_tx.clone(),
            control_rx,
        };

        // Save channels for easy access
        self.beat_tx = Some(beat_tx);
        self.beat_rx = Some(beat_rx);
        self.tick_tx = Some(tick_tx);
        self.tick_rx = Some(tick_rx);
        self.control_tx = Some(control_tx);

        Some(timing_thread)
    }

    /// Stop the clock service and join the timing thread
    pub fn stop_thread(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(sender) = &self.control_tx {
            sender.send(TimingThreadCommand::Shutdown)?;
        }

        if let Some(thread) = self.timing_thread_join.take() {
            thread
                .join()
                .map_err(|_| "Prat: Failed to join timing thread")?;
        }

        self.control_tx = None;
        self.beat_tx = None;

        Ok(())
    }

    /// Set the tempo of the Link session.
    pub fn set_tempo(&self, tempo: f64) -> Result<(), Box<dyn std::error::Error>> {
        self.send_command(TimingThreadCommand::SetLinkTempo(tempo))
    }

    /// Start the clock and Ableton Link transport (if sync is enabled).
    pub fn start_clock(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.send_command(TimingThreadCommand::Start)
    }

    /// Stop the clock and Ableton Link transport (if sync is enabled).
    pub fn stop_clock(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.send_command(TimingThreadCommand::Stop)
    }

    /// Set a callback to be called when a beat event occurs.
    /// Internally, this still uses the beat_rx channel to receive beat events, from the timing thread,
    /// so there is no performance gain.
    /// But if the things you want to do with beat events are simple, this is a quick way to set it up.
    pub fn set_beat_callback<F>(&mut self, callback: F)
    where
        F: Fn(BeatEvent) + Send + 'static,
    {
        self.beat_callback = Some(Box::new(callback));
    }

    /// Used to execute the callback if it exists.
    /// Receives beat events from the timing thread, then calls the callback, passing the beat event.
    pub fn process_beat(&mut self) {
        if let Some(receiver) = &mut self.beat_rx {
            while let Ok(beat_event) = receiver.try_recv() {
                if let Some(ref callback) = self.beat_callback {
                    callback(beat_event);
                }
            }
        }
    }

    /// Send a TimingCommand to the timing thread.
    fn send_command(&self, command: TimingThreadCommand) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(sender) = &self.control_tx {
            sender.send(command)?;
        }
        Ok(())
    }

    /***************** Getters, Setters *****************/

    /// Get a handle to this ClockService that can be cloned and shared.
    /// The handle provides access to sequencer control methods without requiring mutable access.
    pub fn handle(&self) -> TimingThreadControlChannel {
        TimingThreadControlChannel {
            control_tx: self.control_tx.clone(),
        }
    }

    /// Subscribe to beat events.
    /// Returns a receiver that can be used to receive tick events.
    /// The receiver will be empty if the clock service is not running.
    pub fn subscribe_to_beats(&self) -> Option<broadcast::Receiver<BeatEvent>> {
        let Some(beat_tx) = &self.beat_tx else {
            return None;
        };
        Some(beat_tx.subscribe())
    }

    /// Subscribe to tick events.
    /// Returns a receiver that can be used to receive tick events.
    /// The receiver will be empty if the clock service is not running.
    pub fn subscribe_to_ticks(&self) -> Option<broadcast::Receiver<TickEvent>> {
        let Some(tick_tx) = &self.tick_tx else {
            return None;
        };
        Some(tick_tx.subscribe())
    }

    /// Get the current quantum.
    pub fn quantum(&self) -> f64 {
        let session_params = self.session_params.lock().unwrap_or_else(|e| {
            eprintln!(
                "Prat: Session params was poisoned, attempting recovery: {:?}",
                e
            );
            e.into_inner()
        });
        session_params.quantum
    }

    /// Get the current tempo.
    pub fn tempo(&self) -> f64 {
        let session_params = self.session_params.lock().unwrap_or_else(|e| {
            eprintln!(
                "Prat: Session params was poisoned, attempting recovery: {:?}",
                e
            );
            e.into_inner()
        });
        session_params.tempo
    }

    /// Get the current link is playing state.
    pub fn link_is_playing(&self) -> bool {
        let session_params = self.session_params.lock().unwrap_or_else(|e| {
            eprintln!(
                "Prat: Session params was poisoned, attempting recovery: {:?}",
                e
            );
            e.into_inner()
        });
        session_params.link_is_playing
    }

    /// Enable the Ableton Link transport.
    pub fn enable_link(&self, enabled: bool) {
        self.link.enable(enabled);
    }

    /// Returns true if the Ableton Link transport is enabled.
    pub fn link_is_enabled(&self) -> bool {
        self.link.is_enabled()
    }

    /// Returns the number of peers in the Ableton Link session.
    pub fn num_peers(&self) -> u64 {
        self.link.num_peers()
    }
}

impl Drop for ClockService {
    /// Ensure that the clock service is stopped when it goes out of scope.
    fn drop(&mut self) {
        println!("Prat: Dropping ClockService...");
        let _ = self.stop_thread();
        println!("Prat: ClockService stopped");
    }
}

impl TimingThreadControlChannel {
    /// Send a TimingCommand to the timing thread.
    /// This is the internal method used by all public command methods.
    fn send_command(&self, command: TimingThreadCommand) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(sender) = &self.control_tx {
            sender.send(command)?;
        }
        Ok(())
    }

    /// Start clock and transport (if sync is enabled).
    pub fn start_clock(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.send_command(TimingThreadCommand::Start)
    }

    /// Stop clock and transport (if sync is enabled).
    pub fn stop_clock(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.send_command(TimingThreadCommand::Stop)
    }

    /// Change the tempo of the Link session. This will set the tempo for everyone in the session.
    pub fn set_tempo(&self, tempo: f64) -> Result<(), Box<dyn std::error::Error>> {
        self.send_command(TimingThreadCommand::SetLinkTempo(tempo))
    }

    /// Set the quantum.
    pub fn set_quantum(&self, quantum: f64) -> Result<(), Box<dyn std::error::Error>> {
        self.send_command(TimingThreadCommand::SetQuantum(quantum))
    }
}

/// Beat Subdivisions.
/// Each subdivision can be cast into an integer, which allows for indexing into a vector.
/// - 0: Triplet, 1: Whole, 2: Half, 3: Quarter, 4: Eighth, 5: Sixteenth
#[repr(u8)] // allows for indexing by casting into integer
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BeatSubdivision {
    Whole = 1,
    Half = 2,
    Quarter = 3,
    Eighth = 4,
    Sixteenth = 5,
    Triplet = 0,
}

impl BeatSubdivision {
    /// Return the time division multiplier for the subdivision.
    pub fn multiplier(&self) -> f64 {
        match self {
            BeatSubdivision::Whole => 0.25,
            BeatSubdivision::Half => 0.5,
            BeatSubdivision::Quarter => 1.0,
            BeatSubdivision::Eighth => 2.0,
            BeatSubdivision::Sixteenth => 4.0,
            BeatSubdivision::Triplet => 3.0,
        }
    }

    /// Returns a symbol representing the beat subdivision.
    /// - o: whole
    /// - 1/2: half
    /// - ♩: quarter
    /// - ♪: eighth
    /// - ♬: sixteenth
    /// - ♪.: triplet
    pub fn symbol(&self) -> &'static str {
        match self {
            BeatSubdivision::Whole => "o",
            BeatSubdivision::Half => "1/2",
            BeatSubdivision::Quarter => "♩",
            BeatSubdivision::Eighth => "♪",
            BeatSubdivision::Sixteenth => "♬",
            BeatSubdivision::Triplet => "♪.",
        }
    }

    /// Return all beat subdivisions.
    pub fn all() -> &'static [BeatSubdivision] {
        &[
            BeatSubdivision::Whole,
            BeatSubdivision::Half,
            BeatSubdivision::Quarter,
            BeatSubdivision::Eighth,
            BeatSubdivision::Sixteenth,
            BeatSubdivision::Triplet,
        ]
    }

    /// Return the beat subdivisions that can be selected by the user.
    /// ## Deprecated compatibility holdover from when ClockService was part of GameOver
    pub fn selectable() -> &'static [BeatSubdivision] {
        &[
            BeatSubdivision::Quarter,
            BeatSubdivision::Eighth,
            BeatSubdivision::Sixteenth,
            BeatSubdivision::Triplet,
        ]
    }
}

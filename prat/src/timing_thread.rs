//! The process responsible for precise timing via Ableton Link
use crate::clockservice::{BeatEvent, BeatSubdivision, SessionParams, TickEvent};
use rusty_link::{AblLink, SessionState};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex, TryLockError,
    },
    thread::{self},
    time::{Duration, Instant},
};
use tokio::sync::broadcast;

/// Commands that can be sent to the timing thread:
///
/// - Start - start the clock (and transport if link is enabled)
/// - Stop - stop the clock (and transport if link is enabled)
/// - SetTempo(f64) -- set the tempo
/// - SetQuantum(f64) -- set the quantum
/// - FollowLinkTempo(f64) -- follow the tempo of the Link session
/// - Shutdown -- shutdown the clock service
#[derive(Debug, Clone, Copy)]
pub enum TimingThreadCommand {
    Start,
    Stop,
    SetLinkTempo(f64),
    SetQuantum(f64),
    FollowLinkTempo(f64),
    Shutdown,
}

/// The state of the timing thread.
pub struct TimingThreadState {
    link: Arc<AblLink>,
    session_state: SessionState,

    // Shareable State
    session_params: Arc<Mutex<SessionParams>>,
    start_stop_sync_enabled: Arc<AtomicBool>,
    clock_is_running: Arc<AtomicBool>,

    // clock state
    last_subdivision_counts: Vec<u64>,
    last_beat: f64,
    time_since_start: i64,
    start_time: i64,
    last_host_time: i64,
    tick_count: u64,

    // High-resolution timing
    target_interval: i64,
    next_tick_time: i64,
}

pub struct TimingThread {
    // Configuration (from current params)
    pub ppqn: u8,
    pub is_sending_ticks: bool,
    pub debug: bool,

    // State
    pub state: TimingThreadState,

    // Output channels
    pub beat_tx: broadcast::Sender<BeatEvent>,
    pub tick_tx: broadcast::Sender<TickEvent>,

    // Command receiver channel
    pub control_rx: broadcast::Receiver<TimingThreadCommand>,
}

impl TimingThread {
    pub fn run(&mut self) {
        if self.debug {
            println!("Prat worker: Starting TimingThread loop...");
        }

        let mut now;

        loop {
            // First, capture the session state from Ableton Link audio thread
            self.state
                .link
                .capture_audio_session_state(&mut self.state.session_state);

            now = Instant::now();

            // Process commands from the control channel (non-blocking)
            // Audio session state captured above is used in the commands.
            while let Ok(command) = self.control_rx.try_recv() {
                match command {
                    TimingThreadCommand::Start => self.command_start(),
                    TimingThreadCommand::Stop => self.command_stop(),
                    TimingThreadCommand::SetLinkTempo(tempo) => self.command_set_tempo(tempo),
                    TimingThreadCommand::SetQuantum(quantum) => self.command_set_quantum(quantum),
                    TimingThreadCommand::FollowLinkTempo(tempo) => self.follow_tempo(tempo),
                    TimingThreadCommand::Shutdown => return,
                }
            }

            // Load the result of any start/stop commands
            let clock_is_running = self.state.clock_is_running.load(Ordering::SeqCst);

            // Monitor session state changes
            self.update_session_params(clock_is_running);

            // Get the current host time
            self.state.last_host_time = self.state.link.clock_micros();

            // Check tick timing
            if clock_is_running && self.state.last_host_time >= self.state.next_tick_time {
                // Update tick schedule and get tick queue
                let ticks_to_process = self.update_tick_schedule();

                // Emit beat events if any are triggered
                for _ in 0..ticks_to_process {
                    if let Some(beat_event) = self.get_beat_event() {
                        self.emit_beat_event(beat_event);
                    }
                }

                // Emit tick event if enabled
                if self.is_sending_ticks && self.state.last_beat > 0.0 {
                    self.emit_tick_event(now);
                }

                // Update the next tick time
                self.state.next_tick_time += self.state.target_interval;
            }

            // Precise sleep until next tick time
            if clock_is_running {
                let sleep_duration = self
                    .state
                    .next_tick_time
                    .saturating_sub(self.state.last_host_time);
                if sleep_duration > 1000 {
                    thread::sleep(Duration::from_micros(sleep_duration as u64 - 500));
                // Wake slightly early
                } else {
                    //thread::yield_now(); // Yield for very short waits
                    std::hint::spin_loop();
                }
            } else {
                thread::sleep(Duration::from_millis(10)); // Idle sleep when not running
            }
        }
    }

    /// Update the tick schedule and return the number of ticks to process
    fn update_tick_schedule(&mut self) -> i32 {
        // Handle catching up if we're behind (maintain timing grid integrity)
        // This is crucial for MIDI clock stability - we must never break the absolute
        // timing grid as it would cause phase drift and sync issues with other devices
        let mut ticks_to_process = 1;
        let mut next_scheduled_time = self.state.next_tick_time + self.state.target_interval;

        // Check if we need to catch up on multiple ticks
        while next_scheduled_time <= self.state.last_host_time {
            ticks_to_process += 1;
            next_scheduled_time += self.state.target_interval;
        }

        // Log if we had to catch up (indicates system load issues)
        if ticks_to_process > 1 && self.debug {
            eprintln!(
                "\n   Prat worker: (warning) Catching up {} ticks (system may be overloaded)\n",
                ticks_to_process
            );
        }

        ticks_to_process
    }

    /// Emit a beat event (non-blocking)
    fn emit_beat_event(&mut self, beat_event: BeatEvent) {
        // Emit a beat event (non-blocking)
        match self.beat_tx.send(beat_event) {
            Ok(_) => {}
            Err(_) => {
                if self.debug {
                    eprintln!("Prat worker: Beat channel is full, dropping beat event");
                }
            }
        }
    }

    /// Emit a tick event (non-blocking)
    fn emit_tick_event(&mut self, now: Instant) {
        match self.tick_tx.send(TickEvent {
            tick_count: self.state.tick_count,
            tick_time: self.state.last_host_time,
            tick_instant: now,
            next_tick_time: self.state.next_tick_time,
            secs_since_start: (self.state.last_host_time - self.state.start_time) as f64
                / 1_000_000.0,
        }) {
            Ok(_) => {}
            Err(_) => {
                if self.debug {
                    eprintln!("Prat worker: Tick channel is full. Dropping tick event");
                }
            }
        }

        // Increment tick count
        self.state.tick_count += 1;
    }

    /// Detect beat subdivisions and build the BeatEvent
    fn get_beat_event(&mut self) -> Option<BeatEvent> {
        // Read quantum value from Session params
        let session_params = self.state.session_params.lock().unwrap_or_else(|e| {
            eprintln!(
                "Prat worker: Session params was poisoned, attempting recovery: {:?}",
                e
            );
            e.into_inner()
        });
        let quantum = session_params.quantum;
        std::mem::drop(session_params);

        // Get current beat position from Link
        let beats = self
            .state
            .session_state
            .beat_at_time(self.state.last_host_time, quantum);

        // Get current phase from Link
        let phase = self
            .state
            .session_state
            .phase_at_time(self.state.last_host_time, quantum);

        let mut triggered_subdivisions = Vec::new();

        // Check for subdivisions if time has advanced
        if beats > self.state.last_beat {
            // Check each subdivision type
            for subdivision in BeatSubdivision::all() {
                let current_subdivision_time = beats * subdivision.multiplier();
                let current_subdivision_count = current_subdivision_time.floor() as u64;
                let last_subdivision_count =
                    self.state.last_subdivision_counts[*subdivision as usize];

                if current_subdivision_count > last_subdivision_count {
                    triggered_subdivisions.push(*subdivision);
                }

                // Update the stored count
                self.state.last_subdivision_counts[*subdivision as usize] =
                    current_subdivision_count;
            }
        }

        // Update stored state
        self.state.last_beat = beats;
        self.state.time_since_start = self.state.last_host_time - self.state.start_time;

        if !triggered_subdivisions.is_empty() {
            Some(BeatEvent {
                subdivisions: triggered_subdivisions,
                subdivision_counts: self.state.last_subdivision_counts.clone(),
                beat: beats,
                phase,
                secs_since_start: self.state.time_since_start as f64 / 1_000_000.0,
            })
        } else {
            None
        }
    }

    /// Read the Link SessionState and update module SessionParams.
    /// Trigger tempo change or start/stop the clock if needed.
    fn update_session_params(&mut self, clock_is_running: bool) {
        let mut session_params = match self.state.session_params.try_lock() {
            Ok(lock) => lock,
            Err(TryLockError::WouldBlock) => {
                eprintln!(
                    "Prat worker: (warning) Session params lock would block, aborting params update"
                );
                return;
            }
            Err(TryLockError::Poisoned(e)) => {
                eprintln!(
                    "Prat worker: Session params was poisoned, attempting recovery: {:?}",
                    e
                );
                e.into_inner()
            }
        };

        let mut tempo_changed = false;
        let mut link_is_playing_changed = false;

        // Update tempo
        let link_tempo = self.state.session_state.tempo();
        if link_tempo != session_params.tempo {
            session_params.tempo = link_tempo;
            tempo_changed = true;
        }

        // Update Link playing state
        let link_is_playing = self.state.session_state.is_playing();
        if link_is_playing != session_params.link_is_playing {
            session_params.link_is_playing = link_is_playing;
            link_is_playing_changed = true;
        }

        std::mem::drop(session_params);

        // Follow tempo if needed
        if tempo_changed {
            self.follow_tempo(link_tempo);
        }

        // Start/stop the clock if needed
        if link_is_playing_changed && self.state.start_stop_sync_enabled.load(Ordering::SeqCst) {
            if link_is_playing && !clock_is_running {
                println!("Prat worker: Link is playing, starting clock");
                self.command_start();
            } else if !link_is_playing && clock_is_running {
                println!("Prat worker: Link is not playing, stopping clock");
                self.command_stop();
            }
        }
    }

    /******************* Commands *******************/
    /// Start the clock with alignment to the Link session beat and phase, if already running
    fn command_start(&mut self) {
        println!("Prat worker: Received Start command");
        let start_time = self.state.link.clock_micros();
        let num_peers = self.state.link.num_peers();
        println!("Prat worker: Num peers: {}", num_peers);

        let session_params = self.state.session_params.lock().unwrap_or_else(|e| {
            eprintln!(
                "Prat worker: Session params was poisoned, attempting recovery: {:?}",
                e
            );
            e.into_inner()
        });

        let quantum = session_params.quantum;
        std::mem::drop(session_params);

        if num_peers == 0 {
            // No peers: Reset everything to start at beat 0, phase 0
            self.state.start_time = start_time;
            self.state.clock_is_running.store(true, Ordering::SeqCst);

            // When no peers, first tick comes at time = 0.0, so we start at 0.
            self.state.tick_count = 0;

            // Reset subdivision counts and beat tracking
            self.state.last_subdivision_counts = vec![0; BeatSubdivision::all().len()];
            self.state.last_beat = 0.0;

            // Reset the Link session to start at beat 0 immediately
            self.state
                .session_state
                .set_is_playing_and_request_beat_at_time(
                    true, start_time, 0.0, // Start at beat 0
                    quantum,
                );

            // Commit the session state to the Link audio thread
            self.state
                .link
                .commit_audio_session_state(&self.state.session_state);

            // Set the next tick time
            self.state.next_tick_time = start_time;
        } else {
            // Has peers: Quantized launching - wait for next quantum boundary
            // This ensures phase synchronization across all participants
            let current_beat = self.state.session_state.beat_at_time(start_time, quantum);

            // Calculate next quantum boundary for quantized launching
            let next_quantum_beat = (current_beat / quantum).ceil() * quantum;
            let next_quantum_time = self
                .state
                .session_state
                .time_at_beat(next_quantum_beat, quantum);

            // Set the actual start time to the next quantum boundary
            self.state.start_time = next_quantum_time;
            self.state.clock_is_running.store(true, Ordering::SeqCst);

            // When there are peers, first tick starts at start_time + target_interval
            // so we start at 1. Not sure why it's different from no peers.
            // Update: no longer true -- we start at 0. Still needs investigation.
            self.state.tick_count = 0;

            self.state.last_subdivision_counts = vec![0; BeatSubdivision::all().len()];
            self.state.last_beat = 0.0;

            // Don't reset the Link session - maintain existing timeline for phase sync
            self.state
                .session_state
                .set_is_playing_and_request_beat_at_time(true, self.state.start_time, 0.0, quantum);

            // Commit the session state to the Link audio thread
            self.state
                .link
                .commit_audio_session_state(&self.state.session_state);

            // Set the next tick time to the start time (quantum boundary)
            self.state.next_tick_time = self.state.start_time;
        }
    }

    /// Stop the clock
    fn command_stop(&mut self) {
        println!("Prat worker: Received Stop command");

        // Stop the clock locally
        self.state.clock_is_running.store(false, Ordering::SeqCst);

        // Update the Link session to stop playing
        self.state
            .session_state
            .set_is_playing(false, self.state.link.clock_micros());
        self.state
            .link
            .commit_audio_session_state(&self.state.session_state);
    }

    /// Set Session tempo and Prat tick timing based on the new tempo
    fn command_set_tempo(&mut self, tempo: f64) {
        // Use audio thread capture for timing-critical operations
        self.state
            .link
            .capture_audio_session_state(&mut self.state.session_state);
        let host_time = self.state.link.clock_micros();
        self.state.session_state.set_tempo(tempo, host_time);
        self.state
            .link
            .commit_audio_session_state(&self.state.session_state);

        // Update timing interval based on new tempo
        self.state.target_interval = (60_000_000.0 / (tempo * self.ppqn as f64)) as i64;

        if self.debug {
            println!(
                "Prat worker: Session Tempo changed to {:.1} BPM, new interval: {:.3}ms",
                tempo,
                self.state.target_interval as f64 / 1000.0
            );
        }
    }

    /// Set Prat quantum and timing interval based on the new quantum
    fn command_set_quantum(&mut self, new_quantum: f64) {
        let mut session_params = self.state.session_params.lock().unwrap_or_else(|e| {
            eprintln!(
                "Prat worker: Session params was poisoned, attempting recovery: {:?}",
                e
            );
            e.into_inner()
        });
        session_params.quantum = new_quantum;
    }

    /// Update tick interval based on the new tempo
    fn follow_tempo(&mut self, tempo: f64) {
        // Capture the old tick interval
        let old_interval = self.state.target_interval;

        // Update timing interval based on new tempo
        self.state.target_interval = (60_000_000.0 / (tempo * self.ppqn as f64)) as i64;

        // Adjust the next tick time, compensating for the old interval
        self.state.next_tick_time =
            self.state.next_tick_time - old_interval + self.state.target_interval;

        if self.debug {
            println!(
                "Prat worker: Following Session Tempo: {:.1} BPM, new tick interval: {:.3}ms",
                tempo,
                self.state.target_interval as f64 / 1000.0
            );
        }
    }
}

impl TimingThreadState {
    pub fn new(
        link: Arc<AblLink>,
        session_params: Arc<Mutex<SessionParams>>,
        clock_is_running: Arc<AtomicBool>,
        start_stop_sync_enabled: Arc<AtomicBool>,
        ppqn: u8,
        debug: bool,
    ) -> Self {
        let session_params_local = session_params.lock().unwrap_or_else(|e| {
            eprintln!(
                "Prat worker: Session params was poisoned, attempting recovery: {:?}",
                e
            );
            e.into_inner()
        });

        // Calculate initial target_interval based on starting tempo and ppqn
        let initial_target_interval =
            (60_000_000.0 / (session_params_local.tempo * ppqn as f64)) as i64; // ppqn pulses per quarter note

        if debug {
            println!(
                "Prat worker: Initial timing setup - Tempo: {:.1} BPM, PPQN: {}, Interval: {:.3}ms",
                session_params_local.tempo,
                ppqn,
                (initial_target_interval as f64 / 1000.0)
            );
        }

        std::mem::drop(session_params_local);

        Self {
            link,
            session_params: Arc::clone(&session_params),
            session_state: SessionState::new(),
            clock_is_running,
            start_stop_sync_enabled,
            last_subdivision_counts: vec![0; BeatSubdivision::all().len()],
            tick_count: 0,
            last_beat: 0.0,
            time_since_start: 0,
            target_interval: initial_target_interval,
            next_tick_time: initial_target_interval,
            start_time: 0,
            last_host_time: 0,
        }
    }
}

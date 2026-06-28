pub mod clockservice;
pub use clockservice::{BeatSubdivision, ClockService};

mod timing_thread;

use std::thread;
use std::time::Duration;

const USE_CALLBACKS: bool = false;

// Example function showing how to use the ClockService
pub fn example_usage() {
    println!("Prat: Starting ClockService example...");

    let mut clock = ClockService::with()
        .tempo(120.0)
        .quantum(4.0)
        .ppqn(24)
        .enable_ticks()
        .enable_start_stop_sync()
        .debug()
        .build();

    // Demonstrate callback functionality
    if USE_CALLBACKS {
        clock.set_beat_callback(|beat_event| {
                for subdivision in beat_event.subdivisions {
                    match subdivision {
                        BeatSubdivision::Quarter => println!(
                            "🥁 Quarter note at beat {:.2}, count: {}, phase {:.2}, time at beat {:.6}",
                            beat_event.beat,
                            beat_event.subdivision_counts[subdivision as usize],
                            beat_event.phase,
                            beat_event.secs_since_start
                        ),
                        BeatSubdivision::Eighth => println!(
                            "🎵 Eighth note at beat {:.2}, count: {}, phase {:.2}, time at beat {:.6}",
                            beat_event.beat,
                            beat_event.subdivision_counts[subdivision as usize],
                            beat_event.phase,
                            beat_event.secs_since_start
                        ),
                        BeatSubdivision::Sixteenth => {
                            println!(
                                "🎶 Sixteenth note at beat {:.2}, count: {}, phase {:.2}, time at beat {:.6}",
                                beat_event.beat,
                                beat_event.subdivision_counts[subdivision as usize],
                                beat_event.phase,
                                beat_event.secs_since_start
                            )
                        }
                        _ => println!(
                            "⏱️  {} at beat {:.2}, count: {}, phase {:.2}, time at beat {:.6}",
                            subdivision.symbol(),
                            beat_event.beat,
                            beat_event.subdivision_counts[subdivision as usize],
                            beat_event.phase,
                            beat_event.secs_since_start
                        ),
                    }
                }
            });
    }

    // Start the timing thread
    match clock.start_thread() {
        Ok(()) => println!("Prat: ClockService thread started successfully"),
        Err(e) => {
            println!("Prat: Failed to start clock service thread: {}", e);
            return;
        }
    }

    // Subscribe to beat and tick events
    let mut tick_rx = clock.subscribe_to_ticks().unwrap();
    let mut beat_rx = clock.subscribe_to_beats().unwrap();

    // Process beats and ticks for a few seconds
    let _ = clock.start_clock();
    println!("Prat: Start command sent");
    let start_time = std::time::Instant::now();
    while start_time.elapsed() < Duration::from_secs(10) {
        // Process beat events from the subscription
        if !USE_CALLBACKS {
            // Process tick events from the subscription
            while let Ok(tick_event) = tick_rx.try_recv() {
                println!(
                    "Tick #{}: time: {}, next: {}, secs: {:.6}s",
                    tick_event.tick_count,
                    tick_event.tick_time,
                    tick_event.next_tick_time,
                    tick_event.secs_since_start
                );
            }

            // Process beat events from the subscription
            while let Ok(beat_event) = beat_rx.try_recv() {
                for subdivision in beat_event.subdivisions {
                    match subdivision {
                        BeatSubdivision::Quarter => println!(
                            "🥁 Quarter note at beat {:.2}, count: {}, phase {:.2}, time at beat {:.6}",
                            beat_event.beat,
                            beat_event.subdivision_counts[subdivision as usize],
                            beat_event.phase,
                            beat_event.secs_since_start
                        ),
                        BeatSubdivision::Eighth => println!(
                            "🎵 Eighth note at beat {:.2}, count: {}, phase {:.2}, time at beat {:.6}",
                            beat_event.beat,
                            beat_event.subdivision_counts[subdivision as usize],
                            beat_event.phase,
                            beat_event.secs_since_start
                        ),
                        BeatSubdivision::Sixteenth => {
                            println!(
                                "🎶 Sixteenth note at beat {:.2}, count: {}, phase {:.2}, time at beat {:.6}",
                                beat_event.beat,
                                beat_event.subdivision_counts[subdivision as usize],
                                beat_event.phase,
                                beat_event.secs_since_start
                            )
                        }
                        _ => println!(
                            "⏱️  {} at beat {:.2}, count: {}, phase {:.2}, time at beat {:.6}",
                            subdivision.symbol(),
                            beat_event.beat,
                            beat_event.subdivision_counts[subdivision as usize],
                            beat_event.phase,
                            beat_event.secs_since_start
                        ),
                    }
                }
            }
        } else {
            // Callback-based processing: trigger events from within ClockService
            clock.process_beat();
        }

        thread::sleep(Duration::from_millis(1));
    }

    // Stop the clock
    println!("Prat: Stopping clock...");
    clock.stop_clock().expect("Prat: Failed to stop clock");
    println!("Prat: Clock stopped");

    // Wait 5 secs
    thread::sleep(Duration::from_secs(5));

    // Start the clock again
    println!("Prat: Starting clock again...");
    let result = clock.start_clock();
    match result {
        Ok(()) => println!("Prat: Start clock result: {:?}", result),
        Err(e) => println!("Prat: Start clock error: {:?}", e),
    }

    let start_time = std::time::Instant::now();
    while start_time.elapsed() < Duration::from_secs(15) {
        // Process beat events from the subscription
        if !USE_CALLBACKS {
            // Process tick events from the subscription
            while let Ok(tick_event) = tick_rx.try_recv() {
                println!(
                    "Tick #{}: time: {}, next: {}, secs: {:.6}s",
                    tick_event.tick_count,
                    tick_event.tick_time,
                    tick_event.next_tick_time,
                    tick_event.secs_since_start
                );
            }

            // Process beat events from the subscription
            while let Ok(beat_event) = beat_rx.try_recv() {
                for subdivision in beat_event.subdivisions {
                    match subdivision {
                        BeatSubdivision::Quarter => println!(
                            "🥁 Quarter note at beat {:.2}, count: {}, phase {:.2}, time at beat {:.6}",
                            beat_event.beat,
                            beat_event.subdivision_counts[subdivision as usize],
                            beat_event.phase,
                            beat_event.secs_since_start
                        ),
                        BeatSubdivision::Eighth => println!(
                            "🎵 Eighth note at beat {:.2}, count: {}, phase {:.2}, time at beat {:.6}",
                            beat_event.beat,
                            beat_event.subdivision_counts[subdivision as usize],
                            beat_event.phase,
                            beat_event.secs_since_start
                        ),
                        BeatSubdivision::Sixteenth => {
                            println!(
                                "🎶 Sixteenth note at beat {:.2}, count: {}, phase {:.2}, time at beat {:.6}",
                                beat_event.beat,
                                beat_event.subdivision_counts[subdivision as usize],
                                beat_event.phase,
                                beat_event.secs_since_start
                            )
                        }
                        _ => println!(
                            "⏱️  {} at beat {:.2}, count: {}, phase {:.2}, time at beat {:.6}",
                            subdivision.symbol(),
                            beat_event.beat,
                            beat_event.subdivision_counts[subdivision as usize],
                            beat_event.phase,
                            beat_event.secs_since_start
                        ),
                    }
                }
            }
        } else {
            // Callback-based processing: trigger events from within ClockService
            clock.process_beat();
        }

        thread::sleep(Duration::from_millis(1));
    }

    // Change tempo mid-stream
    println!("Prat: Changing tempo to 180 BPM...");

    if let Err(e) = clock.handle().set_tempo(180.0) {
        eprintln!("Prat: Failed to set tempo: {}", e);
    }

    // Run for a few more seconds
    let start_time = std::time::Instant::now();
    while start_time.elapsed() < Duration::from_secs(10) {
        // Continue processing beat events
        while let Ok(beat_event) = beat_rx.try_recv() {
            for subdivision in beat_event.subdivisions {
                println!(
                    "Fast tempo: ⏱️  {} at beat {:.2}, count: {}, phase {:.2}, time at beat {:.6}",
                    subdivision.symbol(),
                    beat_event.beat,
                    beat_event.subdivision_counts[subdivision as usize],
                    beat_event.phase,
                    beat_event.secs_since_start
                );
            }
        }

        // Process tick events from the subscription
        while let Ok(tick_event) = tick_rx.try_recv() {
            println!(
                "Fast tempo tick #{}: {:.6}s",
                tick_event.tick_count, tick_event.secs_since_start
            );
        }

        thread::sleep(Duration::from_millis(1));
    }

    // Stop the clock service
    match clock.stop_thread() {
        Ok(()) => println!("Prat: Clock service stopped successfully"),
        Err(e) => eprintln!("Prat: Error stopping clock service: {}", e),
    }

    println!("Prat: Example completed!");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_clock_service_basic() {
        let mut clock = ClockService::with()
            .tempo(120.0)
            .quantum(4.0)
            .build();

        // Test basic functionality
        assert!(clock.link_is_enabled());

        // Start the clock service
        clock
            .start_thread()
            .expect("Prat: Failed to start clock service");

        // Subscribe to beat events
        let mut beat_rx = clock.subscribe_to_beats().unwrap();

        // Set up a beat counter
        let beat_count = Arc::new(Mutex::new(0));
        let beat_count_clone = Arc::clone(&beat_count);

        // Run for a short time and count beat events
        let start_time = std::time::Instant::now();
        while start_time.elapsed() < Duration::from_millis(500) {
            if let Ok(beat_event) = beat_rx.try_recv() {
                let mut count = beat_count_clone.lock().unwrap();
                *count += beat_event.subdivisions.len();
                println!(
                    "Beat! Subdivisions: {:?}, Beat: {:.2}, Time at beat: {:.5}, Phase: {:.1}",
                    beat_event.subdivisions,
                    beat_event.beat,
                    beat_event.secs_since_start,
                    beat_event.phase
                );
            }
            thread::sleep(Duration::from_millis(1));
        }

        // Stop the clock service
        clock
            .stop_thread()
            .expect("Prat: Failed to stop clock service");

        // Check that we got some beat events
        let final_count = *beat_count.lock().unwrap();
        println!("Prat: Total beat events received: {}", final_count);
        assert!(
            final_count > 0,
            "Prat: Should have received some beat events"
        );
    }
}

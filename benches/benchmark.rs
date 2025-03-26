use std::{fs::create_dir_all, path::Path};

use criterion::{criterion_group, criterion_main, Criterion};

use search_index::{PrefixIndex, Score};

fn bench_prefix_index(c: &mut Criterion) {
    let dir = env!("CARGO_MANIFEST_DIR");
    let base_dir = Path::new(dir).join("benches");
    let data_file = base_dir
        .join("data.tsv")
        .as_os_str()
        .to_str()
        .expect("Invalid path")
        .to_string();
    let index_dir = base_dir
        .join("index")
        .as_os_str()
        .to_str()
        .expect("Invalid path")
        .to_string();

    // create index dir if it doesn't exist
    create_dir_all(&index_dir).expect("Failed to create index dir");

    println!("Building index at {index_dir} from {data_file}");

    PrefixIndex::build(&data_file, &index_dir, true).expect("Failed to build index");

    let index = PrefixIndex::load(&data_file, &index_dir).expect("Failed to load index");

    c.bench_function("find_matches_1", |b| {
        b.iter(|| {
            let _ = index
                .find_matches(
                    "the united states",
                    Score::Occurrence,
                    0.0,
                    0.0,
                    None,
                    false,
                )
                .expect("Failed to find matches");
        })
    });

    c.bench_function("find_matches_1_min_4", |b| {
        b.iter(|| {
            let _ = index
                .find_matches(
                    "the united states",
                    Score::Occurrence,
                    0.0,
                    0.0,
                    Some(4),
                    false,
                )
                .expect("Failed to find matches");
        })
    });

    c.bench_function("find_matches_2", |b| {
        b.iter(|| {
            let _ = index
                .find_matches("angela m", Score::Occurrence, 0.0, 0.0, None, false)
                .expect("Failed to find matches");
        })
    });

    c.bench_function("find_matches_2_no_ref", |b| {
        b.iter(|| {
            let _ = index
                .find_matches("angela m", Score::Occurrence, 0.0, 0.0, None, true)
                .expect("Failed to find matches");
        })
    });
}

criterion_group!(benches, bench_prefix_index);
criterion_main!(benches);

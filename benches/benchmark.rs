use std::{fs::create_dir_all, path::Path};

use criterion::{criterion_group, criterion_main, Criterion};

use search_index::{IndexData, Mapping, PrefixIndex, Score};

fn bench_data_and_mapping(c: &mut Criterion) {
    let dir = env!("CARGO_MANIFEST_DIR");
    let base_dir = Path::new(dir).join("benches");
    let data_file = base_dir
        .join("data.tsv")
        .as_os_str()
        .to_str()
        .expect("Invalid path")
        .to_string();
    let offsets_file = base_dir
        .join("data.offsets")
        .as_os_str()
        .to_str()
        .expect("Invalid path")
        .to_string();
    let mapping_file = base_dir
        .join("mapping.bin")
        .as_os_str()
        .to_str()
        .expect("Invalid path")
        .to_string();

    IndexData::build(&data_file, &offsets_file).expect("Failed to load data");
    let data = IndexData::load(&data_file, &offsets_file).expect("Failed to load data");

    let mut g = c.benchmark_group("data_and_mapping");

    g.bench_function("build_data", |b| {
        b.iter(|| {
            IndexData::build(&data_file, &offsets_file).expect("Failed to load data");
        })
    });

    g.bench_function("load_data", |b| {
        b.iter(|| {
            let _ = IndexData::load(&data_file, &offsets_file).expect("Failed to load data");
        })
    });

    g.bench_function("get_val", |b| {
        b.iter(|| {
            let _ = data
                .get_val(data.len() / 2, 3)
                .expect("Failed to get value");
        })
    });

    g.bench_function("get_row", |b| {
        b.iter(|| {
            let _ = data.get_row(data.len() / 2).expect("Failed to get row");
        })
    });

    g.bench_function("build_mapping", |b| {
        b.iter(|| {
            let _ =
                Mapping::build(data.clone(), &mapping_file, 3).expect("Failed to build mapping");
        })
    });

    g.finish();
}

fn bench_prefix_index(c: &mut Criterion) {
    let dir = env!("CARGO_MANIFEST_DIR");
    let base_dir = Path::new(dir).join("benches");
    let data_file = base_dir
        .join("data.tsv")
        .as_os_str()
        .to_str()
        .expect("Invalid path")
        .to_string();
    let offsets_file = base_dir
        .join("data.offsets")
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

    IndexData::build(&data_file, &offsets_file).expect("Failed to build index data");
    let data = IndexData::load(&data_file, &offsets_file).expect("Failed to load index data");

    PrefixIndex::build(data.clone(), &index_dir, true).expect("Failed to build index");

    let index = PrefixIndex::load(data, &index_dir).expect("Failed to load index");

    let mut g = c.benchmark_group("prefix_index");

    g.bench_function("find_matches_1", |b| {
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

    g.bench_function("find_matches_1_min_4", |b| {
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

    g.bench_function("find_matches_2", |b| {
        b.iter(|| {
            let _ = index
                .find_matches("angela m", Score::Occurrence, 0.0, 0.0, None, false)
                .expect("Failed to find matches");
        })
    });

    g.bench_function("find_matches_2_no_ref", |b| {
        b.iter(|| {
            let _ = index
                .find_matches("angela m", Score::Occurrence, 0.0, 0.0, None, true)
                .expect("Failed to find matches");
        })
    });

    g.finish();
}

criterion_group!(benches, bench_prefix_index, bench_data_and_mapping);
criterion_main!(benches);

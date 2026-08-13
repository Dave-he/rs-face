#[test]
fn read_pgm_test() {
    use std::path::Path;
    let result = rs_face::read_pgm(Path::new("/tmp/rs-face-test/frames/face_00001.pgm"));
    println!("{:?}", result.as_ref().map(|i| (i.w, i.h, i.data.len())));
    assert!(result.is_ok());
}

use std::cmp::Ordering;
use std::error;
use std::fmt;
use std::fs::File;
use std::io::BufReader;
use std::num::ParseIntError;
use std::path::Path;
use std::str::FromStr;

const NO_CHILD: i8 = -1;
const NO_DATA: u64 = 0;
const LEAF: [i8; 8] = [NO_CHILD; 8];

#[derive(Clone, Copy, Debug, Default)]
pub struct Node {
    pub data: u64,
    pub children_base: u64,
    pub children_offsets: [i8; 8],
}

impl Node {
    pub fn new() -> Self {
        Node {
            data: 0,
            children_base: 0,
            children_offsets: [0; 8],
        }
    }

    pub fn has_child(&self, idx: usize) -> bool {
        self.children_offsets[idx] != NO_CHILD
    }

    pub fn is_leaf(&self) -> bool {
        self.children_offsets == LEAF
    }

    pub fn has_data(&self) -> bool {
        self.data != NO_DATA
    }

    pub fn is_null(&self) -> bool {
        self.is_leaf() && !self.has_data()
    }

    pub fn child_count(&self) -> u8 {
        self.children_offsets
            .iter()
            .fold(0, |acc, x| acc + if *x == NO_CHILD { 0 } else { 1 })
    }

    pub fn child_pos(&self, idx: usize) -> Option<u64> {
        match self.children_offsets[idx] {
            NO_CHILD => None,
            o => Some(self.children_base + o as u64),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct VoxelData {
    pub morton: u64,
    pub color: [f32; 3],
    pub normal: [f32; 3],
}

impl VoxelData {
    pub fn new(morton: u64, normal: [f32; 3], color: [f32; 3]) -> Self {
        VoxelData {
            morton,
            normal,
            color,
        }
    }
}

impl Eq for VoxelData {}

impl Ord for VoxelData {
    fn cmp(&self, other: &Self) -> Ordering {
        self.morton.cmp(&other.morton)
    }
}

impl PartialOrd for VoxelData {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for VoxelData {
    fn eq(&self, other: &Self) -> bool {
        self.morton == other.morton
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OctreeInfo {
    pub version: i32,
    pub gridlength: u32,
    pub n_nodes: u64,
    pub n_data: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OctreeInfoError {
    UnsupportedVersion,
    UnrecognizedKeyword(String),
    GridLengthError,
    NodeCountError(ParseIntError),
    DataCountError(ParseIntError),
    MissingField(String),
    Parse(String),
}

impl fmt::Display for OctreeInfoError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            OctreeInfoError::UnsupportedVersion => {
                write!(f, "Only version 1 is supported currently")
            }
            OctreeInfoError::UnrecognizedKeyword(ref k) => write!(f, "Unrecognized keyword: {}", k),
            OctreeInfoError::GridLengthError => {
                write!(f, "Only gridlengths of power 2 are supported")
            }
            OctreeInfoError::NodeCountError(ref e) | OctreeInfoError::DataCountError(ref e) => {
                e.fmt(f)
            }
            OctreeInfoError::Parse(ref e) => write!(f, "Parse: {}", e),
            OctreeInfoError::MissingField(ref e) => write!(f, "Missing Field: {}", e),
        }
    }
}

impl error::Error for OctreeInfoError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        // not implemented
        None
    }
}

impl FromStr for OctreeInfo {
    type Err = OctreeInfoError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut gridlength: Option<u32> = None;
        let mut n_nodes: Option<u64> = None;
        let mut n_data: Option<u64> = None;

        let mut lines = s.lines();
        // first line has to be the '#octreeheader' with the version
        let version = match lines.next() {
            Some(s) => {
                let mut header = s.split_whitespace();
                match header.next() {
                    Some("#octreeheader") => match header.next() {
                        Some("1") => 1,
                        _ => return Err(OctreeInfoError::UnsupportedVersion),
                    },
                    _ => return Err(OctreeInfoError::Parse("Invalid header".to_owned())),
                }
            }
            _ => return Err(OctreeInfoError::Parse("Invalid header".to_owned())),
        };
        for l in lines {
            let mut words = l.split_whitespace();
            match words.next() {
                Some(key) => {
                    match key {
                        "gridlength" => {
                            gridlength = match words.next() {
                                Some(l) => match l.parse::<u32>() {
                                    Ok(l) => {
                                        if (l & (l - 1)) == 0 {
                                            Some(l)
                                        } else {
                                            return Err(OctreeInfoError::GridLengthError);
                                        }
                                    }
                                    Err(_) => return Err(OctreeInfoError::GridLengthError),
                                },
                                _ => return Err(OctreeInfoError::GridLengthError),
                            }
                        }
                        "n_nodes" => {
                            n_nodes = match words.next() {
                                Some(l) => match l.parse::<u64>() {
                                    Ok(c) => Some(c),
                                    Err(e) => return Err(OctreeInfoError::NodeCountError(e)),
                                },
                                _ => {
                                    return Err(OctreeInfoError::Parse(
                                        "missing parameter for n_nodes".to_owned(),
                                    ))
                                }
                            }
                        }
                        "n_data" => {
                            n_data = match words.next() {
                                Some(l) => match l.parse::<u64>() {
                                    Ok(c) => Some(c),
                                    Err(e) => return Err(OctreeInfoError::DataCountError(e)),
                                },
                                _ => {
                                    return Err(OctreeInfoError::Parse(
                                        "missing parameter for n_data".to_owned(),
                                    ))
                                }
                            }
                        }
                        "END" => break,
                        k => return Err(OctreeInfoError::UnrecognizedKeyword(k.to_string())),
                    };
                }
                None => return Err(OctreeInfoError::Parse("Unexpected line ending".to_owned())),
            }
        }

        let gridlength = match gridlength {
            Some(g) => g,
            None => return Err(OctreeInfoError::MissingField("gridlength".to_owned())),
        };
        let n_nodes = match n_nodes {
            Some(n) => n,
            None => return Err(OctreeInfoError::MissingField("n_nodes".to_owned())),
        };
        let n_data = match n_data {
            Some(d) => d,
            None => return Err(OctreeInfoError::MissingField("n_data".to_owned())),
        };

        Ok(OctreeInfo {
            version,
            gridlength,
            n_nodes,
            n_data,
        })
    }
}

pub struct OctreeFile {
    info: OctreeInfo,
    node_file: BufReader<File>,
    data_file: BufReader<File>,
}

#[derive(Debug)]
pub enum OctreeFileError {
    OctreeInfoError(OctreeInfoError),
    IOError(std::io::Error),
    TODO,
}

impl fmt::Display for OctreeFileError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            _ => write!(f, "TODO"),
        }
    }
}

impl error::Error for OctreeFileError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        // not implemented
        None
    }
}

impl From<OctreeInfoError> for OctreeFileError {
    fn from(err: OctreeInfoError) -> OctreeFileError {
        OctreeFileError::OctreeInfoError(err)
    }
}

impl From<std::io::Error> for OctreeFileError {
    fn from(err: std::io::Error) -> OctreeFileError {
        OctreeFileError::IOError(err)
    }
}

impl OctreeFile {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<OctreeFile, OctreeFileError> {
        // let path = Path
        let base_filename = match path.as_ref().file_stem() {
            Some(p) => p,
            None => return Err(OctreeFileError::TODO),
        };
        let path = match path.as_ref().parent() {
            Some(p) => p,
            None => return Err(OctreeFileError::TODO),
        };
        // probably TODO remove unwraps of to_str
        let filename = path.join(format!("{}.octree", base_filename.to_str().unwrap()));
        let node_filename = path.join(format!("{}.octreenodes", base_filename.to_str().unwrap()));
        let data_filename = path.join(format!("{}.octreedata", base_filename.to_str().unwrap()));

        let info = std::fs::read_to_string(filename)?.parse()?;
        let node_file = BufReader::new(File::open(node_filename)?);
        let data_file = BufReader::new(File::open(data_filename)?);

        Ok(OctreeFile {
            info,
            node_file,
            data_file,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parse_octree_info() {
        let octree_info: Result<OctreeInfo, OctreeInfoError> = "#octreeheader 1
gridlength 64
n_nodes 5542
n_data 4206
END"
        .parse();
        assert_eq!(
            octree_info,
            Ok(OctreeInfo {
                version: 1,
                gridlength: 64,
                n_nodes: 5542,
                n_data: 4206
            })
        );

        let octree_info: Result<OctreeInfo, OctreeInfoError> = "#octreeheader 1
gridlength
n_nodes 5542
n_data 4206
END"
        .parse();
        assert_eq!(octree_info, Err(OctreeInfoError::GridLengthError));

        let octree_info: Result<OctreeInfo, OctreeInfoError> = "#octreeheader 1
gridlength 64
n_nodes 5542
END
"
        .parse();
        assert_eq!(
            octree_info,
            Err(OctreeInfoError::MissingField("n_data".to_owned()))
        );
    }

    #[test]
    fn open_octree_file() {
        eprintln!("exec dir: {:?}", std::env::current_dir().unwrap());
        let octree_file = OctreeFile::open("./testdata/shroom164_1.octree").expect("file should exist");

        assert_eq!(
            octree_file.info,
            OctreeInfo {
                version: 1,
                gridlength: 64,
                n_nodes: 5542,
                n_data: 4206
            }
        );
    }
}

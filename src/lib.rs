use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fs::File;
use std::fs::OpenOptions;
use std::io;
use std::io::{Read, Seek, SeekFrom};
use std::mem::size_of;
use std::num::ParseIntError;
use std::path::Path;
use std::slice::from_raw_parts_mut;
use std::str::FromStr;
use thiserror::Error;

pub const NO_CHILD: u8 = 0xff;
pub const NO_DATA: u64 = 0;
pub const LEAF: [u8; 8] = [NO_CHILD; 8];

unsafe fn any_as_u8_slice_mut<T: Sized>(p: &mut T) -> &mut [u8] {
    from_raw_parts_mut((p as *mut T) as *mut u8, size_of::<T>())
}

unsafe fn any_slice_as_u8_slice_mut<T: Sized>(p: &mut [T]) -> &mut [u8] {
    from_raw_parts_mut((p as *mut [T]) as *mut u8, p.len() * size_of::<T>())
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Node {
    pub data: u64,
    pub children_base: u64,
    pub children_offsets: [u8; 8],
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

    pub fn child_count(&self) -> usize {
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

#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum OctreeInfoError {
    #[error("Unsupported version, only version 1 is supported currently")]
    UnsupportedVersion,
    #[error("Unrecognized keyword: {0}")]
    UnrecognizedKeyword(String),
    #[error("Error while parsing gridlength, only gridlengths of power 2 are supported")]
    GridLengthError,
    #[error("Error while parsing value of gridlength, n_nodes or n_data: {0}")]
    ParseIntValue(#[from] ParseIntError),
    #[error("Missing field: {0}")]
    MissingField(String),
    #[error("Missing value for: {0}")]
    MissingValue(String),
    #[error("Error while parsing: {0}")]
    Parse(String),
}

impl FromStr for OctreeInfo {
    type Err = OctreeInfoError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use OctreeInfoError as OErr;

        let mut lines = s.lines();
        // first line has to be the '#octreeheader' with the version
        let header = lines
            .next()
            .ok_or_else(|| OErr::Parse("Invalid header".to_owned()))?
            .split_whitespace()
            .collect::<Vec<_>>();

        let version = match header.as_slice() {
            ["#octreeheader", "1"] => Ok(1),
            ["#octreeheader", _] => Err(OErr::UnsupportedVersion),
            _ => Err(OErr::Parse("Invalid header".into())),
        }?;

        let mut key_values = BTreeMap::new();

        for l in lines {
            let split_line = l.split_whitespace().collect::<Vec<_>>();
            match split_line.as_slice() {
                [k @ ("gridlength" | "n_nodes" | "n_data"), v] => {
                    let v = v.parse::<u64>()?;
                    key_values.insert(k.clone(), v);
                }
                ["END"] => break,
                [k @ ("gridlength" | "n_nodes" | "n_data")] => {
                    return Err(OErr::MissingValue(k.to_string()))
                }
                [k, ..] => return Err(OErr::UnrecognizedKeyword(k.to_string())),
                [] => return Err(OErr::Parse("Unexpected line ending".to_owned())),
            };
        }

        let gridlength = key_values
            .remove(&"gridlength")
            .ok_or_else(|| OErr::MissingField("gridlength".to_owned()))?
            .try_into()
            .map_err(|_| OErr::GridLengthError)?;
        // check if power of 2
        if !((gridlength != 0) && ((gridlength & (gridlength - 1)) == 0)) {
            return Err(OErr::GridLengthError);
        }
        let n_nodes = key_values
            .remove(&"n_nodes")
            .ok_or_else(|| OErr::MissingField("n_nodes".to_owned()))?;
        let n_data = key_values
            .remove(&"n_data")
            .ok_or_else(|| OErr::MissingField("n_data".to_owned()))?;

        Ok(OctreeInfo {
            version,
            gridlength,
            n_nodes,
            n_data,
        })
    }
}

pub struct OctreeFile {
    #[allow(dead_code)]
    pub info: OctreeInfo,
    // TODO do some smart buffering, BufReader is not well suited due to the vast number of seeks
    // (which can often be random)
    node_file: File,
    data_file: File,
}

#[derive(Error, Debug)]
pub enum OctreeFileError {
    #[error("Error while reading octree header: {0}")]
    OctreeHeaderError(#[from] OctreeInfoError),
    #[error("Error while reading file: {0}")]
    IOError(#[from] io::Error),
}

impl OctreeFile {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<OctreeFile, OctreeFileError> {
        use io::ErrorKind as Kind;
        let path = path.as_ref();
        let base_filename = path
            .file_stem()
            .ok_or_else(|| io::Error::new(Kind::NotFound, "The path doesn't contain a file"))?;
        let path = path.parent().ok_or_else(|| {
            io::Error::new(Kind::InvalidData, "The path is the root instead of a file")
        })?;
        let base_filename = base_filename.to_str().ok_or_else(|| {
            io::Error::new(Kind::InvalidData, "filename contains invalid unicode")
        })?;
        let filename = path.join(format!("{}.octree", base_filename));
        let node_filename = path.join(format!("{}.octreenodes", base_filename));
        let data_filename = path.join(format!("{}.octreedata", base_filename));

        let info = std::fs::read_to_string(filename)?.parse()?;
        let node_file = OpenOptions::new().read(true).open(node_filename)?;
        let data_file = OpenOptions::new().read(true).open(data_filename)?;

        Ok(OctreeFile {
            info,
            node_file,
            data_file,
        })
    }

    pub fn read_root_node(&mut self) -> Result<Node, OctreeFileError> {
        // TODO compare with nom parsing, specifically bench the two solutions, also think of
        // endianess
        self.node_file
            .seek(SeekFrom::End(-(size_of::<Node>() as i64)))?;
        let mut node = Node::new();
        self.node_file
            .read_exact(unsafe { any_as_u8_slice_mut(&mut node) })?;
        Ok(node)
    }

    pub fn read_all_nodes(&mut self) -> Result<Vec<Node>, OctreeFileError> {
        self.node_file.seek(SeekFrom::Start(0))?;
        let length = self.node_file.metadata()?.len();

        if length != size_of::<Node>() as u64 * self.info.n_nodes {
            return Err(OctreeFileError::IOError(io::Error::new(
                io::ErrorKind::InvalidData,
                "count of node differs in nodes file",
            )));
        }

        let mut nodes: Vec<Node> = vec![Node::new(); self.info.n_nodes as usize];
        self.node_file
            .read_exact(unsafe { any_slice_as_u8_slice_mut(nodes.as_mut_slice()) })?;
        Ok(nodes)
    }

    pub fn read_children_nodes(&mut self, node: &Node) -> Result<Vec<Node>, OctreeFileError> {
        let children_address = node.children_base;
        self.node_file.seek(SeekFrom::Start(
            (size_of::<Node>() as u64) * children_address,
        ))?;
        let mut children_nodes: Vec<Node> = vec![Node::new(); node.child_count()];

        self.node_file
            .read_exact(unsafe { any_slice_as_u8_slice_mut(children_nodes.as_mut_slice()) })?;

        Ok(children_nodes)
    }

    pub fn read_voxel_data(&mut self, node: &Node) -> Result<VoxelData, OctreeFileError> {
        if node.data == NO_DATA {
            return Err(OctreeFileError::IOError(io::Error::new(
                io::ErrorKind::InvalidInput,
                "The node doesn't have any data",
            )));
        }
        self.data_file
            .seek(SeekFrom::Start((size_of::<VoxelData>() as u64) * node.data))?;
        let mut data = VoxelData::new(0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        self.data_file
            .read_exact(unsafe { any_as_u8_slice_mut(&mut data) })?;

        Ok(data)
    }

    pub fn read_all_voxel_data(&mut self) -> Result<Vec<VoxelData>, OctreeFileError> {
        self.data_file.seek(SeekFrom::Start(0))?;
        let length = self.data_file.metadata()?.len();

        if length != size_of::<VoxelData>() as u64 * self.info.n_data {
            return Err(OctreeFileError::IOError(io::Error::new(
                io::ErrorKind::InvalidData,
                "count of voxel data differs in data file",
            )));
        }

        let mut voxel_data: Vec<VoxelData> =
            vec![VoxelData::new(0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]); self.info.n_data as usize];
        self.data_file
            .read_exact(unsafe { any_slice_as_u8_slice_mut(voxel_data.as_mut_slice()) })?;
        Ok(voxel_data)
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
        assert_eq!(
            octree_info,
            Err(OctreeInfoError::MissingValue("gridlength".into()))
        );

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
        let octree_file =
            OctreeFile::open("./testdata/shroom164_1.octree").expect("file should exist");

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

    #[test]
    fn read_root_node() {
        let mut octree_file =
            OctreeFile::open("./testdata/shroom164_1.octree").expect("file should exist");
        let root_node_read = octree_file.read_root_node().unwrap();
        let mut root_node_expect = Node::new();
        root_node_expect.data = NO_DATA;
        root_node_expect.children_base = 5534;
        root_node_expect.children_offsets = [NO_CHILD, 0, 1, 2, 3, 4, 5, 6];
        assert_eq!(root_node_read, root_node_expect);
    }

    #[test]
    fn read_children_nodes() {
        let mut octree_file =
            OctreeFile::open("./testdata/shroom164_1.octree").expect("file should exist");
        let root_node = octree_file.read_root_node().unwrap();
        let children_nodes = octree_file.read_children_nodes(&root_node).unwrap();
        assert_eq!(
            children_nodes,
            [
                Node {
                    data: 0,
                    children_base: 25,
                    children_offsets: [
                        NO_CHILD, NO_CHILD, NO_CHILD, NO_CHILD, NO_CHILD, NO_CHILD, 0, NO_CHILD
                    ]
                },
                Node {
                    data: 0,
                    children_base: 1025,
                    children_offsets: [
                        NO_CHILD, NO_CHILD, NO_CHILD, NO_CHILD, NO_CHILD, 0, NO_CHILD, 1
                    ]
                },
                Node {
                    data: 0,
                    children_base: 2068,
                    children_offsets: [
                        NO_CHILD, NO_CHILD, NO_CHILD, NO_CHILD, 0, NO_CHILD, 1, NO_CHILD
                    ]
                },
                Node {
                    data: 0,
                    children_base: 2201,
                    children_offsets: [
                        NO_CHILD, 0, NO_CHILD, 1, NO_CHILD, NO_CHILD, NO_CHILD, NO_CHILD
                    ]
                },
                Node {
                    data: 0,
                    children_base: 2832,
                    children_offsets: [
                        0, NO_CHILD, 1, NO_CHILD, NO_CHILD, NO_CHILD, NO_CHILD, NO_CHILD
                    ]
                },
                Node {
                    data: 0,
                    children_base: 4224,
                    children_offsets: [
                        NO_CHILD, 0, NO_CHILD, 1, NO_CHILD, NO_CHILD, NO_CHILD, NO_CHILD
                    ]
                },
                Node {
                    data: 0,
                    children_base: 5532,
                    children_offsets: [
                        0, NO_CHILD, 1, NO_CHILD, NO_CHILD, NO_CHILD, NO_CHILD, NO_CHILD
                    ]
                },
            ]
        );
    }

    #[test]
    fn count_all_nodes_and_voxel_data() {
        fn count_all_nodes_rec(node: &Node, file: &mut OctreeFile) -> (usize, usize) {
            let node_with_data = if node.data == NO_DATA { 0 } else { 1 };
            if node.child_count() == 0 {
                (1, node_with_data)
            } else {
                let (child_node_count, child_data_count) = file
                    .read_children_nodes(node)
                    .unwrap()
                    .iter()
                    .fold((0, 0), |acc, ref n| {
                        let (c, d) = count_all_nodes_rec(n, file);
                        (acc.0 + c, acc.1 + d)
                    });
                (child_node_count + 1, child_data_count + node_with_data)
            }
        }
        let mut octree_file =
            OctreeFile::open("./testdata/shroom164_1.octree").expect("file should exist");
        let root_node = octree_file.read_root_node().unwrap();
        let counts = count_all_nodes_rec(&root_node, &mut octree_file);
        // for some reason the first data block in the data file is not used thus -1
        assert_eq!(
            counts,
            (
                octree_file.info.n_nodes as usize,
                octree_file.info.n_data as usize - 1
            )
        );
    }

    #[test]
    fn read_voxel_data() {
        let mut octree_file =
            OctreeFile::open("./testdata/shroom164_1.octree").expect("file should exist");
        // second node in node file
        let mut node = Node::new();
        node.data = 2;
        node.children_base = 0;
        node.children_offsets = [NO_CHILD; 8];
        let data = octree_file.read_voxel_data(&node).unwrap();

        assert_eq!(
            data,
            VoxelData::new(
                60725,
                [0.0, 0.0, 0.0],
                [0.105000392, -0.0576205514, -0.992801487]
            )
        )
    }

    #[test]
    fn read_all_nodes() {
        let mut octree_file =
            OctreeFile::open("./testdata/shroom164_1.octree").expect("file should exist");
        let nodes = octree_file.read_all_nodes().unwrap();
        assert_eq!(nodes.len(), octree_file.info.n_nodes as usize);
    }

    #[test]
    fn read_all_voxel_data() {
        let mut octree_file =
            OctreeFile::open("./testdata/shroom164_1.octree").expect("file should exist");
        let voxel_data = octree_file.read_all_voxel_data().unwrap();
        assert_eq!(voxel_data.len(), octree_file.info.n_data as usize);
    }
}

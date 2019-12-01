use nom::{
    bytes::complete::tag,
    character::complete::{digit1, line_ending, space0, space1},
    combinator::map_res,
    error::{convert_error, ParseError},
    sequence::{preceded, terminated, tuple},
    IResult,
};
use std::cmp::Ordering;
use std::error;
use std::fmt;
use std::fs::File;
use std::fs::OpenOptions;
use std::io;
use std::io::{Read, Seek, SeekFrom};
use std::mem::size_of;
use std::path::Path;
use std::slice::from_raw_parts_mut;
use std::str::FromStr;

const NO_CHILD: i8 = -1;
const NO_DATA: u64 = 0;
const LEAF: [i8; 8] = [NO_CHILD; 8];

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

impl OctreeInfo {
    fn parse_from_header_str<'a, E: ParseError<&'a str>>(
        i: &'a str,
    ) -> IResult<&'a str, OctreeInfo, E> {
        fn parse_line<'b, T: FromStr, E: ParseError<&'b str>>(
            t: &'b str,
        ) -> impl Fn(&'b str) -> IResult<&'b str, T, E> {
            move |i: &'b str| {
                preceded(
                    tag(t),
                    terminated(
                        preceded(space1, map_res(digit1, |s: &'b str| s.parse::<T>())),
                        preceded(space0, line_ending),
                    ),
                )(i)
            }
        }
        let (s, (version, gridlength, n_nodes, n_data)): (&str, (i32, u32, u64, u64)) =
            terminated(
                tuple((
                    parse_line("#octreeheader"),
                    parse_line("gridlength"),
                    parse_line("n_nodes"),
                    parse_line("n_data"),
                )),
                tag("END"),
            )(i)?;

        Ok((
            s,
            OctreeInfo {
                version,
                gridlength,
                n_nodes,
                n_data,
            },
        ))
    }
}

pub struct OctreeFile {
    #[allow(dead_code)]
    info: OctreeInfo,
    // TODO do some smart buffering, BufReader is not well suited due to the vast number of seeks
    // (which can often be random)
    node_file: File,
    data_file: File,
}

#[derive(Debug)]
pub enum OctreeFileError {
    OctreeHeaderError(String),
    IOError(io::Error),
}

impl fmt::Display for OctreeFileError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            OctreeFileError::OctreeHeaderError(ref s) => {
                write!(f, "Error while reading octree header: {}", s)
            }
            OctreeFileError::IOError(ref e) => write!(f, "Error while reading file: {}", e),
        }
    }
}

impl error::Error for OctreeFileError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        // not implemented
        None
    }
}

impl From<io::Error> for OctreeFileError {
    fn from(err: io::Error) -> OctreeFileError {
        OctreeFileError::IOError(err)
    }
}

impl OctreeFile {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<OctreeFile, OctreeFileError> {
        // let path = Path
        let base_filename = match path.as_ref().file_stem() {
            Some(p) => p,
            None => {
                return Err(OctreeFileError::IOError(io::Error::new(
                    io::ErrorKind::NotFound,
                    "The path doesn't contain a file",
                )))
            }
        };
        let path = match path.as_ref().parent() {
            Some(p) => p,
            None => {
                return Err(OctreeFileError::IOError(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "The path is the root instead of a file",
                )))
            }
        };
        let base_filename = match base_filename.to_str() {
            Some(s) => s,
            None => {
                return Err(OctreeFileError::IOError(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "filename contains invalid unicode",
                )))
            }
        };
        let filename = path.join(format!("{}.octree", base_filename));
        let node_filename = path.join(format!("{}.octreenodes", base_filename));
        let data_filename = path.join(format!("{}.octreedata", base_filename));

        let info_str = std::fs::read_to_string(filename)?;
        let info = match OctreeInfo::parse_from_header_str(info_str.as_ref()) {
            Ok((_, o)) => o,
            Err(nom::Err::Error(e)) | Err(nom::Err::Failure(e)) => {
                return Err(OctreeFileError::OctreeHeaderError(convert_error(
                    info_str.as_ref(),
                    e,
                )))
            }
            // doesn't happen...
            _ => return Err(OctreeFileError::OctreeHeaderError(String::new())),
        };
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use nom::error::VerboseError;

    #[test]
    fn parse_octree_info() {
        let octree_info = OctreeInfo::parse_from_header_str::<VerboseError<&str>>(
            "#octreeheader 1
gridlength 64
n_nodes 5542
n_data 4206
END",
        );
        assert_eq!(
            octree_info,
            Ok((
                "",
                OctreeInfo {
                    version: 1,
                    gridlength: 64,
                    n_nodes: 5542,
                    n_data: 4206
                }
            ))
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
}

use core::borrow::Borrow;
use rand::prelude::*;
use rand::distributions::Uniform;
use std::collections::HashMap;
use std::f64::EPSILON;


pub type Node = Vec<f64>;

pub fn train(nodes: &[Node], cluster_count: usize, max_rounds: u64) -> Option<Vec<Node>> {
    if nodes.len() < cluster_count { return None };

    let mut std_len = 0;
    for (i, node) in nodes.iter().enumerate() {
        let cur_len = node.len();
        if i > 0 && node.len()!= std_len { return None };
        std_len = cur_len;
    };

    let mut centroids = Vec::with_capacity(cluster_count);

    let mut rng = thread_rng();
    for _ in 0..cluster_count {
        let n = nodes.get(rng.sample(Uniform::new(0, nodes.len())));
        centroids.push(n.unwrap().to_owned());
    };
    train2(nodes, max_rounds, centroids)
}

fn train2(nodes: &[Node], max_rounds: u64, mut centroids: Vec<Node>) -> Option<Vec<Node>> {

    let mut movement = true;

    for _ in 0..max_rounds {
        if !movement { break }
        movement = false;

        let mut groups :HashMap<usize,Vec<Node>> = HashMap::new();

        for node in nodes.iter() {
            let near = nearest(node, centroids.borrow());
            groups.entry(near).
                and_modify(|e| e.push(node.to_vec())).
                or_insert_with(|| vec![node.to_vec()]);
        };

        for (key, group) in groups {
            let new_node = mean_node(group);

            if !equal(centroids.get(key).unwrap(), new_node.borrow()) {
                centroids[key] = new_node;
                movement = true
            }
        };
    };


    Some(centroids)
}

fn equal(node1: &Node, node2: &Node) -> bool {
    if node1.len() != node2.len() { return false }

    for (i, j) in node1.iter().zip(node2.iter()) {
        if (i - j).abs() > EPSILON {
            return false
        }
    }
    true
}

pub fn nearest(in_node: &Node, nodes: &[Node]) -> usize {
    let count = nodes.len();

    let mut results :Node = Vec::with_capacity(count);

    nodes.iter().
        for_each(|node| results.push(distance(in_node.borrow(), node)));

    let mut mindex = 0;
    let mut curdist = results.get(0).unwrap();

    for (i, dist) in results.iter().enumerate() {
        if dist < curdist {
            curdist = dist;
            mindex = i;
        }
    };

    mindex
}

fn distance(node1: &Node, node2: &Node) -> f64 {
    let mut squares = Vec::with_capacity(node1.len());

    node1.iter().zip(node2).for_each(|(i, j)| squares.push((i-j).powf(2.0)) );

    squares.iter().sum()
}

fn mean_node(values: Vec<Node>) -> Node {
    let mut new_node :Node = vec![0.0; values.first().unwrap().len()];
    values.iter().for_each(|value| {
        new_node.iter_mut().
            enumerate().
            for_each(|(j, v)| *v += value.get(j).unwrap())
    });
    new_node.iter_mut().for_each(|value| *value /= values.len() as f64);
    new_node
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_works() {
       let observations = vec![
           vec![20.0, 20.0, 20.0, 20.0],
           vec![21.0, 21.0, 21.0, 21.0],
           vec![100.5, 100.5, 100.5, 100.5],
           vec![50.1, 50.1, 50.1, 50.1],
           vec![64.2, 64.2, 64.2, 64.2],
       ];

        let b = observations.clone();
        let centroids = train(&observations, 2, 50);
        let c = centroids.clone();
        if let Some(c) = centroids {
            for centroid in c {
                println!("centroid: {:?}", centroid);
            }
        }

        println!("...");
        for observation in b.iter() {
            let index = nearest(observation.borrow(), c.clone().unwrap().borrow());
            println!("{:?} belongs in cluster {:?}", observation, index+1);
        }
    }
}

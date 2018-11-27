package com.paritosh.weka;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.M5P;
import weka.clusterers.HierarchicalClusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args ) throws Exception
    {
        BufferedReader reader = new BufferedReader(new FileReader("absentieesim_workload_avg.arff"));
        Instances data = new Instances(reader);
        data.deleteAttributeAt(0);
        SimpleKMeans kmeans = new SimpleKMeans();
        kmeans.setNumClusters(5);
     
        kmeans.setMaxIterations(1000);
        kmeans.setPreserveInstancesOrder(true);
     // Perform K-Means clustering.
        try {  
            kmeans.buildClusterer(data);
        } catch (Exception ex) {
            System.err.println("Unable to buld Clusterer: " + ex.getMessage());
            ex.printStackTrace();
        }

        // print out the cluster centroids
        Instances centroids = kmeans.getClusterCentroids();
        
//        kmeans.
        for (int i = 0; i < 5; i++) {
            System.out.print("Cluster " + i + " size: " + kmeans.getClusterSizes()[i]);
            System.out.println(" Centroid: " + centroids.instance(i));
        }
        
        
    }
}

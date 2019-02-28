import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.util.pair.IntFloatPair;
import spark.*;
import javax.servlet.*;
import javax.servlet.http.*;
import java.io.*;
import java.nio.file.*;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import static spark.Spark.*;
import static spark.debug.DebugScreen.*;

import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.FImage;
import org.openimaj.io.IOUtils;
import org.openimaj.ml.annotation.ScoredAnnotation;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;

public class Main {

    private static String TRAINING_IMAGES_PATH_PART0 = "/Users/paulochang/Downloads/sketch-to-code/dataset_part0";
    private static String TRAINER_DATA_FILE_PATH_PART0 = "/Users/paulochang/Downloads/trainer_part0.dat";

    private static String TRAINING_IMAGES_PATH_PART1 = "/Users/paulochang/Downloads/sketch-to-code/dataset_part1";
    private static String TRAINER_DATA_FILE_PATH_PART1 = "/Users/paulochang/Downloads/trainer_part2.dat";

    private static String TRAINING_IMAGES_PATH_PART2 = "/Users/paulochang/Downloads/sketch-to-code/dataset_part2";
    private static String TRAINER_DATA_FILE_PATH_PART2 = "/Users/paulochang/Downloads/trainer_part1.dat";



    public static void main(String[] args) {
        enableDebugScreen();

        File uploadDir = new File("upload");
        uploadDir.mkdir(); // create the upload directory if it doesn't exist

        staticFiles.externalLocation("upload");

        get("/header", (req, res) ->
                "<form method='post' enctype='multipart/form-data'>" // note the enctype
                        + "    <input type='file' name='uploaded_file' accept='.png'>" // make sure to call getPart using the same "name" in the post
                        + "    <button>Upload picture</button>"
                        + "</form>"
        );

        post("/header", (req, res) -> {

            Path tempFile = Files.createTempFile(uploadDir.toPath(), "", "");

            req.attribute("org.eclipse.jetty.multipartConfig", new MultipartConfigElement("/temp"));



            LiblinearAnnotator<FImage, String> trainer = null;
            File inputDataFile = new File(TRAINER_DATA_FILE_PATH);
            if (inputDataFile.isFile()) {
                trainer = IOUtils.readFromFile(inputDataFile);
            } else
            {
                VFSGroupDataset<FImage> allData = null;
                allData = new VFSGroupDataset<FImage>(
                        TRAINING_IMAGES_PATH,
                        ImageUtilities.FIMAGE_READER);

                GroupedDataset<String, ListDataset<FImage>, FImage> data =
                        GroupSampler.sample(allData, 4, false);

                GroupedRandomSplitter<String, FImage> splits =
                        new GroupedRandomSplitter<String, FImage>(data, 2, 0, 2); // 15 training, 15 testing


                DenseSIFT denseSIFT = new DenseSIFT(5, 7);
                PyramidDenseSIFT<FImage> pyramidDenseSIFT = new PyramidDenseSIFT<FImage>(denseSIFT, 6f, 7);

                GroupedDataset<String, ListDataset<FImage>, FImage> sample =
                        GroupedUniformRandomisedSampler.sample(splits.getTrainingDataset(), 2);

                HardAssigner<byte[], float[], IntFloatPair> assigner = trainQuantiser(sample, pyramidDenseSIFT);

                FeatureExtractor<DoubleFV, FImage> extractor = new PHOWExtractor(pyramidDenseSIFT, assigner);

                //
                // Now we’re ready to construct and train a classifier
                //
                trainer = new LiblinearAnnotator<FImage, String>(
                        extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);

                Date start = new Date();
                System.out.println("Classifier training: start");
                trainer.train(splits.getTrainingDataset());
                File f = new File(TRAINER_DATA_FILE_PATH);
                if (!f.getParentFile().exists())
                    f.getParentFile().mkdirs();
                if (!f.exists())
                    f.createNewFile();

                IOUtils.writeToFile(trainer, new File(TRAINER_DATA_FILE_PATH));
                System.out.println("Classifier training: end");
                Date end = new Date();
                long durationSec = (end.getTime() - start.getTime()) / 1000;
                System.out.println("Classifier training duration: " + durationSec + " seconds");
            }

            try (InputStream input = req.raw().getPart("uploaded_file").getInputStream()) { // getPart needs to use same "name" as input field in form
                Files.copy(input, tempFile, StandardCopyOption.REPLACE_EXISTING);


            FImage query = ImageUtilities.readF(tempFile.toFile());

            final List<ScoredAnnotation<String>> scoredAnnotations = trainer.annotate(query);
            final ClassificationResult<String> classificationResult = trainer.classify(query);
            System.out.println("scoredAnnotations: " + scoredAnnotations);
            System.out.println("classificationResult: " + classificationResult.getPredictedClasses());
            return "<h1>Scored annotations :<h1> <p>" + scoredAnnotations +"</p> <h1>classificationResult :<h1> <p>"+classificationResult.getPredictedClasses()+"</p>";

            }

        });

    }

    // methods used for logging
    private static void logInfo(Request req, Path tempFile) throws IOException, ServletException {
        System.out.println("Uploaded file '" + getFileName(req.raw().getPart("uploaded_file")) + "' saved as '" + tempFile.toAbsolutePath() + "'");
    }

    private static String getFileName(Part part) {
        for (String cd : part.getHeader("content-disposition").split(";")) {
            if (cd.trim().startsWith("filename")) {
                return cd.substring(cd.indexOf('=') + 1).trim().replace("\"", "");
            }
        }
        return null;
    }

    static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(
            Dataset<FImage> sample,
            PyramidDenseSIFT<FImage> pyramidDenseSIFT)
    {
        System.out.println("trainQuantiser: start");
        Date start = new Date();
        List<LocalFeatureList<ByteDSIFTKeypoint>> allKeys = new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();

        int i = 0;

        int total = sample.numInstances();
        for (FImage rec : sample) {
            i++;
            System.out.println(String.format("Analysing image %d out of %d", i, total));
            FImage img = rec.getImage();

            pyramidDenseSIFT.analyseImage(img);
            allKeys.add(pyramidDenseSIFT.getByteKeypoints(0.005f));
        }
        final int numberOfDenseSiftFeaturesToExtract = 10000;
        final int numberOfClassesInCluster = 300;
        if (allKeys.size() > numberOfDenseSiftFeaturesToExtract)
            allKeys = allKeys.subList(0, numberOfDenseSiftFeaturesToExtract);

        ByteKMeans km = ByteKMeans.createKDTreeEnsemble(numberOfClassesInCluster);
        DataSource<byte[]> dataSource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(allKeys);
        System.out.println(String.format(
                "Clustering %d image features into %d classes...",
                numberOfDenseSiftFeaturesToExtract, numberOfClassesInCluster));
        ByteCentroidsResult result = km.cluster(dataSource);
        Date end = new Date();
        System.out.println("trainQuantiser: end");
        System.out.println("trainQuantiser duration: " + (end.getTime() - start.getTime())/1000 + " seconds");
        return result.defaultHardAssigner();
    }

    static class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage> {
        PyramidDenseSIFT<FImage> pdsift;
        HardAssigner<byte[], float[], IntFloatPair> assigner;

        public PHOWExtractor(PyramidDenseSIFT<FImage> pdsift, HardAssigner<byte[], float[], IntFloatPair> assigner)
        {
            this.pdsift = pdsift;
            this.assigner = assigner;
        }

        public DoubleFV extractFeature(FImage object) {
            FImage image = object.getImage();
            pdsift.analyseImage(image);

            BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<byte[]>(assigner);

            BlockSpatialAggregator<byte[], SparseIntFV> spatial = new BlockSpatialAggregator<byte[], SparseIntFV>(
                    bovw, 2, 2);

            return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
        }
    }
}
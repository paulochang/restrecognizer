import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.*;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.processing.edges.SUSANEdgeDetector;
import org.openimaj.io.IOUtils;
import org.openimaj.ml.annotation.ScoredAnnotation;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.util.pair.IntFloatPair;
import spark.Request;
import spark.Route;

import javax.servlet.MultipartConfigElement;
import javax.servlet.ServletException;
import javax.servlet.http.Part;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import static spark.Spark.*;
import static spark.debug.DebugScreen.enableDebugScreen;

public class Main {


    private static final String BASE_PATH = "/Users/paulochang/Downloads/sketch-to-code";
    private static final String IMAGES_PATH = BASE_PATH + "/images";
    private static final String DAT_FILE_PATH = BASE_PATH + "/training_data_files";
    private static final File UPLOAD_DIRECTORY = new File("upload");

    private static final Route formRoute = (req, res) ->
            "<form method='post' enctype='multipart/form-data'>" // note the enctype
                    + "    <input type='file' name='uploaded_file' accept='.png'>" // make sure to call getPart using the same "name" in the post
                    + "    <button>Upload picture</button>"
                    + "</form>";

    //Part0: numGroup=2, numTraining=3, numTesting=3, number=5
    //Part1: numGroup=3, numTraining=3, numTesting=3, number=7
    //Part2: numGroup=3, numTraining=6, numTesting=6, number=12

    private static final String ROUTE_PART0 = "/header";
    private static final String TRAINING_IMAGES_PATH_PART0 = IMAGES_PATH + "/dataset_part0";
    private static final String TRAINER_DATA_FILE_PATH_PART0 = DAT_FILE_PATH + "/trainer_part0.dat";
    private static final int PART0_NUMGROUP = 2;
    private static final int PART0_NUMTRAINING = 3;
    private static final int PART0_NUMTESTING = 3;
    private static final int PART0_NUMBER = 5;


    private static final String ROUTE_PART1 = "/stage";
    private static final String TRAINING_IMAGES_PATH_PART1 = IMAGES_PATH + "/dataset_part1";
    private static final String TRAINER_DATA_FILE_PATH_PART1 = DAT_FILE_PATH + "/trainer_part1.dat";
    private static final int PART1_NUMGROUP = 3;
    private static final int PART1_NUMTRAINING = 3;
    private static final int PART1_NUMTESTING = 3;
    private static final int PART1_NUMBER = 7;

    private static final String ROUTE_PART2 = "/tesaserlist";
    private static final String TRAINING_IMAGES_PATH_PART2 = IMAGES_PATH + "/dataset_part1";
    private static final String TRAINER_DATA_FILE_PATH_PART2 = DAT_FILE_PATH + "/trainer_part1.dat";
    private static final int PART2_NUMGROUP = 3;
    private static final int PART2_NUMTRAINING = 6;
    private static final int PART2_NUMTESTING = 6;
    private static final int PART2_NUMBER = 12;

    public static final ConfigObject[] CONFIG_ARRAYS = {
            new ConfigObject(ROUTE_PART0,
                    TRAINING_IMAGES_PATH_PART0,
                    TRAINER_DATA_FILE_PATH_PART0,
                    PART0_NUMGROUP,
                    PART0_NUMTRAINING,
                    PART0_NUMTESTING,
                    PART0_NUMBER),

            new ConfigObject(ROUTE_PART1,
                    TRAINING_IMAGES_PATH_PART1,
                    TRAINER_DATA_FILE_PATH_PART1,
                    PART1_NUMGROUP,
                    PART1_NUMTRAINING,
                    PART1_NUMTESTING,
                    PART1_NUMBER),

            new ConfigObject(ROUTE_PART2,
                    TRAINING_IMAGES_PATH_PART2,
                    TRAINER_DATA_FILE_PATH_PART2,
                    PART2_NUMGROUP,
                    PART2_NUMTRAINING,
                    PART2_NUMTESTING,
                    PART2_NUMBER),
    };



    public static void main(String[] args) {
        enableDebugScreen();

        UPLOAD_DIRECTORY.mkdir(); // create the upload directory if it doesn't exist

        staticFiles.externalLocation("upload");

        for (int i = 0; i < CONFIG_ARRAYS.length; i++) {
            setupRoutes(CONFIG_ARRAYS[i]);
        }

    }

    private static void setupRoutes(ConfigObject currentConfig) {
        get(currentConfig.getRoute(), formRoute
        );

        post(currentConfig.getRoute(), (req, res) -> {

            Path tempFile = Files.createTempFile(UPLOAD_DIRECTORY.toPath(), "", "");

            req.attribute("org.eclipse.jetty.multipartConfig", new MultipartConfigElement("/temp"));


            LiblinearAnnotator<FImage, String> trainer = null;
            File inputDataFile = new File(currentConfig.getTrainerDataFile());                                           //TRAINER_DATA_FILE_PATH_PART0
            if (inputDataFile.isFile()) {
                trainer = IOUtils.readFromFile(inputDataFile);
            } else {
                VFSGroupDataset<FImage> allData = null;
                allData = new VFSGroupDataset<FImage>(
                        currentConfig.getTrainingImagesPath(),                                                                 //TRAINING_IMAGES_PATH_PART0
                        ImageUtilities.FIMAGE_READER);

                GroupedDataset<String, ListDataset<FImage>, FImage> data =
                        GroupSampler.sample(allData, currentConfig.getNumGroup(), false);

                GroupedRandomSplitter<String, FImage> splits =
                        new GroupedRandomSplitter<String, FImage>(data, currentConfig.getNumTraining(), 0, currentConfig.getNumTesting()); // 15 training, 15 testing


                DenseSIFT denseSIFT = new DenseSIFT(5, 7);
                PyramidDenseSIFT<FImage> pyramidDenseSIFT = new PyramidDenseSIFT<FImage>(denseSIFT, 6f, 7);

                GroupedDataset<String, ListDataset<FImage>, FImage> sample =
                        GroupedUniformRandomisedSampler.sample(splits.getTrainingDataset(), currentConfig.getNumber());

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
                    File f = new File(currentConfig.getTrainerDataFile());                                                //TRAINER_DATA_FILE_PATH_PART0
                if (!f.getParentFile().exists())
                    f.getParentFile().mkdirs();
                if (!f.exists())
                    f.createNewFile();

                IOUtils.writeToFile(trainer, new File(currentConfig.getTrainerDataFile()));                               //TRAINER_DATA_FILE_PATH_PART0
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
                return "<h1>Scored annotations :<h1> <p>" + scoredAnnotations + "</p> <h1>classificationResult :<h1> <p>" + classificationResult.getPredictedClasses() + "</p>";

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
            PyramidDenseSIFT<FImage> pyramidDenseSIFT) {
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
        System.out.println("trainQuantiser duration: " + (end.getTime() - start.getTime()) / 1000 + " seconds");
        return result.defaultHardAssigner();
    }

    static class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage> {
        PyramidDenseSIFT<FImage> pdsift;
        HardAssigner<byte[], float[], IntFloatPair> assigner;

        public PHOWExtractor(PyramidDenseSIFT<FImage> pdsift, HardAssigner<byte[], float[], IntFloatPair> assigner) {
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
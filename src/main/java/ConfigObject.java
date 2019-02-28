public class ConfigObject {
    private String route;
    private String trainingImagesPath;
    private String trainerDataFile;
    private int numGroup;
    private int numTraining;
    private int numTesting;
    private int number;

    public ConfigObject(String route, String trainingImagesPath, String trainerDataFile) {
        this.route = route;
        this.trainingImagesPath = trainingImagesPath;
        this.trainerDataFile = trainerDataFile;
    }

    public ConfigObject(String route, String trainingImagesPath, String trainerDataFile, int numGroup, int numTraining, int numTesting, int number) {
        this.route = route;
        this.trainingImagesPath = trainingImagesPath;
        this.trainerDataFile = trainerDataFile;
        this.numGroup = numGroup;
        this.numTraining = numTraining;
        this.numTesting = numTesting;
        this.number = number;
    }

    public String getRoute() {
        return route;
    }

    public String getTrainingImagesPath() {
        return trainingImagesPath;
    }

    public String getTrainerDataFile() {
        return trainerDataFile;
    }

    public int getNumGroup() {
        return numGroup;
    }

    public int getNumTraining() {
        return numTraining;
    }

    public int getNumTesting() {
        return numTesting;
    }

    public int getNumber() {
        return number;
    }
}

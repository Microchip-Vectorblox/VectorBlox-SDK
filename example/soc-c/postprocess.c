#include "postprocess.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>


char *lpr_chinese_characters[71] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
         "Anhui", "Beijing", "Chongqing", "Fujian",
         "Gansu", "Guangdong", "Guangxi", "Guizhou",
         "Hainan", "Hebei", "Heilongjiang", "Henan",
         "HongKong", "Hubei", "Hunan", "InnerMongolia",
         "Jiangsu", "Jiangxi", "Jilin", "Liaoning",
         "Macau", "Ningxia", "Qinghai", "Shaanxi",
         "Shandong", "Shanghai", "Shanxi", "Sichuan",
         "Tianjin", "Tibet", "Xinjiang", "Yunnan",
         "Zhejiang", "Police",
         "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
         "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
         "U", "V", "W", "X", "Y", "Z", " "};


char *lpr_characters[37] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
         "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
         "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
         "U", "V", "W", "X", "Y", "Z", " "};


char *imagenet_classes[1000] = {
	"tench",
	"goldfish",
	"great white shark",
	"tiger shark",
	"hammerhead",
	"electric ray",
	"stingray",
	"cock",
	"hen",
	"ostrich",
	"brambling",
	"goldfinch",
	"house finch",
	"junco",
	"indigo bunting",
	"robin",
	"bulbul",
	"jay",
	"magpie",
	"chickadee",
	"water ouzel",
	"kite",
	"bald eagle",
	"vulture",
	"great grey owl",
	"European fire salamander",
	"common newt",
	"eft",
	"spotted salamander",
	"axolotl",
	"bullfrog",
	"tree frog",
	"tailed frog",
	"loggerhead",
	"leatherback turtle",
	"mud turtle",
	"terrapin",
	"box turtle",
	"banded gecko",
	"common iguana",
	"American chameleon",
	"whiptail",
	"agama",
	"frilled lizard",
	"alligator lizard",
	"Gila monster",
	"green lizard",
	"African chameleon",
	"Komodo dragon",
	"African crocodile",
	"American alligator",
	"triceratops",
	"thunder snake",
	"ringneck snake",
	"hognose snake",
	"green snake",
	"king snake",
	"garter snake",
	"water snake",
	"vine snake",
	"night snake",
	"boa constrictor",
	"rock python",
	"Indian cobra",
	"green mamba",
	"sea snake",
	"horned viper",
	"diamondback",
	"sidewinder",
	"trilobite",
	"harvestman",
	"scorpion",
	"black and gold garden spider",
	"barn spider",
	"garden spider",
	"black widow",
	"tarantula",
	"wolf spider",
	"tick",
	"centipede",
	"black grouse",
	"ptarmigan",
	"ruffed grouse",
	"prairie chicken",
	"peacock",
	"quail",
	"partridge",
	"African grey",
	"macaw",
	"sulphur-crested cockatoo",
	"lorikeet",
	"coucal",
	"bee eater",
	"hornbill",
	"hummingbird",
	"jacamar",
	"toucan",
	"drake",
	"red-breasted merganser",
	"goose",
	"black swan",
	"tusker",
	"echidna",
	"platypus",
	"wallaby",
	"koala",
	"wombat",
	"jellyfish",
	"sea anemone",
	"brain coral",
	"flatworm",
	"nematode",
	"conch",
	"snail",
	"slug",
	"sea slug",
	"chiton",
	"chambered nautilus",
	"Dungeness crab",
	"rock crab",
	"fiddler crab",
	"king crab",
	"American lobster",
	"spiny lobster",
	"crayfish",
	"hermit crab",
	"isopod",
	"white stork",
	"black stork",
	"spoonbill",
	"flamingo",
	"little blue heron",
	"American egret",
	"bittern",
	"crane",
	"limpkin",
	"European gallinule",
	"American coot",
	"bustard",
	"ruddy turnstone",
	"red-backed sandpiper",
	"redshank",
	"dowitcher",
	"oystercatcher",
	"pelican",
	"king penguin",
	"albatross",
	"grey whale",
	"killer whale",
	"dugong",
	"sea lion",
	"Chihuahua",
	"Japanese spaniel",
	"Maltese dog",
	"Pekinese",
	"Shih-Tzu",
	"Blenheim spaniel",
	"papillon",
	"toy terrier",
	"Rhodesian ridgeback",
	"Afghan hound",
	"basset",
	"beagle",
	"bloodhound",
	"bluetick",
	"black-and-tan coonhound",
	"Walker hound",
	"English foxhound",
	"redbone",
	"borzoi",
	"Irish wolfhound",
	"Italian greyhound",
	"whippet",
	"Ibizan hound",
	"Norwegian elkhound",
	"otterhound",
	"Saluki",
	"Scottish deerhound",
	"Weimaraner",
	"Staffordshire bullterrier",
	"American Staffordshire terrier",
	"Bedlington terrier",
	"Border terrier",
	"Kerry blue terrier",
	"Irish terrier",
	"Norfolk terrier",
	"Norwich terrier",
	"Yorkshire terrier",
	"wire-haired fox terrier",
	"Lakeland terrier",
	"Sealyham terrier",
	"Airedale",
	"cairn",
	"Australian terrier",
	"Dandie Dinmont",
	"Boston bull",
	"miniature schnauzer",
	"giant schnauzer",
	"standard schnauzer",
	"Scotch terrier",
	"Tibetan terrier",
	"silky terrier",
	"soft-coated wheaten terrier",
	"West Highland white terrier",
	"Lhasa",
	"flat-coated retriever",
	"curly-coated retriever",
	"golden retriever",
	"Labrador retriever",
	"Chesapeake Bay retriever",
	"German short-haired pointer",
	"vizsla",
	"English setter",
	"Irish setter",
	"Gordon setter",
	"Brittany spaniel",
	"clumber",
	"English springer",
	"Welsh springer spaniel",
	"cocker spaniel",
	"Sussex spaniel",
	"Irish water spaniel",
	"kuvasz",
	"schipperke",
	"groenendael",
	"malinois",
	"briard",
	"kelpie",
	"komondor",
	"Old English sheepdog",
	"Shetland sheepdog",
	"collie",
	"Border collie",
	"Bouvier des Flandres",
	"Rottweiler",
	"German shepherd",
	"Doberman",
	"miniature pinscher",
	"Greater Swiss Mountain dog",
	"Bernese mountain dog",
	"Appenzeller",
	"EntleBucher",
	"boxer",
	"bull mastiff",
	"Tibetan mastiff",
	"French bulldog",
	"Great Dane",
	"Saint Bernard",
	"Eskimo dog",
	"malamute",
	"Siberian husky",
	"dalmatian",
	"affenpinscher",
	"basenji",
	"pug",
	"Leonberg",
	"Newfoundland",
	"Great Pyrenees",
	"Samoyed",
	"Pomeranian",
	"chow",
	"keeshond",
	"Brabancon griffon",
	"Pembroke",
	"Cardigan",
	"toy poodle",
	"miniature poodle",
	"standard poodle",
	"Mexican hairless",
	"timber wolf",
	"white wolf",
	"red wolf",
	"coyote",
	"dingo",
	"dhole",
	"African hunting dog",
	"hyena",
	"red fox",
	"kit fox",
	"Arctic fox",
	"grey fox",
	"tabby",
	"tiger cat",
	"Persian cat",
	"Siamese cat",
	"Egyptian cat",
	"cougar",
	"lynx",
	"leopard",
	"snow leopard",
	"jaguar",
	"lion",
	"tiger",
	"cheetah",
	"brown bear",
	"American black bear",
	"ice bear",
	"sloth bear",
	"mongoose",
	"meerkat",
	"tiger beetle",
	"ladybug",
	"ground beetle",
	"long-horned beetle",
	"leaf beetle",
	"dung beetle",
	"rhinoceros beetle",
	"weevil",
	"fly",
	"bee",
	"ant",
	"grasshopper",
	"cricket",
	"walking stick",
	"cockroach",
	"mantis",
	"cicada",
	"leafhopper",
	"lacewing",
	"dragonfly",
	"damselfly",
	"admiral",
	"ringlet",
	"monarch",
	"cabbage butterfly",
	"sulphur butterfly",
	"lycaenid",
	"starfish",
	"sea urchin",
	"sea cucumber",
	"wood rabbit",
	"hare",
	"Angora",
	"hamster",
	"porcupine",
	"fox squirrel",
	"marmot",
	"beaver",
	"guinea pig",
	"sorrel",
	"zebra",
	"hog",
	"wild boar",
	"warthog",
	"hippopotamus",
	"ox",
	"water buffalo",
	"bison",
	"ram",
	"bighorn",
	"ibex",
	"hartebeest",
	"impala",
	"gazelle",
	"Arabian camel",
	"llama",
	"weasel",
	"mink",
	"polecat",
	"black-footed ferret",
	"otter",
	"skunk",
	"badger",
	"armadillo",
	"three-toed sloth",
	"orangutan",
	"gorilla",
	"chimpanzee",
	"gibbon",
	"siamang",
	"guenon",
	"patas",
	"baboon",
	"macaque",
	"langur",
	"colobus",
	"proboscis monkey",
	"marmoset",
	"capuchin",
	"howler monkey",
	"titi",
	"spider monkey",
	"squirrel monkey",
	"Madagascar cat",
	"indri",
	"Indian elephant",
	"African elephant",
	"lesser panda",
	"giant panda",
	"barracouta",
	"eel",
	"coho",
	"rock beauty",
	"anemone fish",
	"sturgeon",
	"gar",
	"lionfish",
	"puffer",
	"abacus",
	"abaya",
	"academic gown",
	"accordion",
	"acoustic guitar",
	"aircraft carrier",
	"airliner",
	"airship",
	"altar",
	"ambulance",
	"amphibian",
	"analog clock",
	"apiary",
	"apron",
	"ashcan",
	"assault rifle",
	"backpack",
	"bakery",
	"balance beam",
	"balloon",
	"ballpoint",
	"Band Aid",
	"banjo",
	"bannister",
	"barbell",
	"barber chair",
	"barbershop",
	"barn",
	"barometer",
	"barrel",
	"barrow",
	"baseball",
	"basketball",
	"bassinet",
	"bassoon",
	"bathing cap",
	"bath towel",
	"bathtub",
	"beach wagon",
	"beacon",
	"beaker",
	"bearskin",
	"beer bottle",
	"beer glass",
	"bell cote",
	"bib",
	"bicycle-built-for-two",
	"bikini",
	"binder",
	"binoculars",
	"birdhouse",
	"boathouse",
	"bobsled",
	"bolo tie",
	"bonnet",
	"bookcase",
	"bookshop",
	"bottlecap",
	"bow",
	"bow tie",
	"brass",
	"brassiere",
	"breakwater",
	"breastplate",
	"broom",
	"bucket",
	"buckle",
	"bulletproof vest",
	"bullet train",
	"butcher shop",
	"cab",
	"caldron",
	"candle",
	"cannon",
	"canoe",
	"can opener",
	"cardigan",
	"car mirror",
	"carousel",
	"carpenter's kit",
	"carton",
	"car wheel",
	"cash machine",
	"cassette",
	"cassette player",
	"castle",
	"catamaran",
	"CD player",
	"cello",
	"cellular telephone",
	"chain",
	"chainlink fence",
	"chain mail",
	"chain saw",
	"chest",
	"chiffonier",
	"chime",
	"china cabinet",
	"Christmas stocking",
	"church",
	"cinema",
	"cleaver",
	"cliff dwelling",
	"cloak",
	"clog",
	"cocktail shaker",
	"coffee mug",
	"coffeepot",
	"coil",
	"combination lock",
	"computer keyboard",
	"confectionery",
	"container ship",
	"convertible",
	"corkscrew",
	"cornet",
	"cowboy boot",
	"cowboy hat",
	"cradle",
	"crane",
	"crash helmet",
	"crate",
	"crib",
	"Crock Pot",
	"croquet ball",
	"crutch",
	"cuirass",
	"dam",
	"desk",
	"desktop computer",
	"dial telephone",
	"diaper",
	"digital clock",
	"digital watch",
	"dining table",
	"dishrag",
	"dishwasher",
	"disk brake",
	"dock",
	"dogsled",
	"dome",
	"doormat",
	"drilling platform",
	"drum",
	"drumstick",
	"dumbbell",
	"Dutch oven",
	"electric fan",
	"electric guitar",
	"electric locomotive",
	"entertainment center",
	"envelope",
	"espresso maker",
	"face powder",
	"feather boa",
	"file",
	"fireboat",
	"fire engine",
	"fire screen",
	"flagpole",
	"flute",
	"folding chair",
	"football helmet",
	"forklift",
	"fountain",
	"fountain pen",
	"four-poster",
	"freight car",
	"French horn",
	"frying pan",
	"fur coat",
	"garbage truck",
	"gasmask",
	"gas pump",
	"goblet",
	"go-kart",
	"golf ball",
	"golfcart",
	"gondola",
	"gong",
	"gown",
	"grand piano",
	"greenhouse",
	"grille",
	"grocery store",
	"guillotine",
	"hair slide",
	"hair spray",
	"half track",
	"hammer",
	"hamper",
	"hand blower",
	"hand-held computer",
	"handkerchief",
	"hard disc",
	"harmonica",
	"harp",
	"harvester",
	"hatchet",
	"holster",
	"home theater",
	"honeycomb",
	"hook",
	"hoopskirt",
	"horizontal bar",
	"horse cart",
	"hourglass",
	"iPod",
	"iron",
	"jack-o'-lantern",
	"jean",
	"jeep",
	"jersey",
	"jigsaw puzzle",
	"jinrikisha",
	"joystick",
	"kimono",
	"knee pad",
	"knot",
	"lab coat",
	"ladle",
	"lampshade",
	"laptop",
	"lawn mower",
	"lens cap",
	"letter opener",
	"library",
	"lifeboat",
	"lighter",
	"limousine",
	"liner",
	"lipstick",
	"Loafer",
	"lotion",
	"loudspeaker",
	"loupe",
	"lumbermill",
	"magnetic compass",
	"mailbag",
	"mailbox",
	"maillot",
	"maillot",
	"manhole cover",
	"maraca",
	"marimba",
	"mask",
	"matchstick",
	"maypole",
	"maze",
	"measuring cup",
	"medicine chest",
	"megalith",
	"microphone",
	"microwave",
	"military uniform",
	"milk can",
	"minibus",
	"miniskirt",
	"minivan",
	"missile",
	"mitten",
	"mixing bowl",
	"mobile home",
	"Model T",
	"modem",
	"monastery",
	"monitor",
	"moped",
	"mortar",
	"mortarboard",
	"mosque",
	"mosquito net",
	"motor scooter",
	"mountain bike",
	"mountain tent",
	"mouse",
	"mousetrap",
	"moving van",
	"muzzle",
	"nail",
	"neck brace",
	"necklace",
	"nipple",
	"notebook",
	"obelisk",
	"oboe",
	"ocarina",
	"odometer",
	"oil filter",
	"organ",
	"oscilloscope",
	"overskirt",
	"oxcart",
	"oxygen mask",
	"packet",
	"paddle",
	"paddlewheel",
	"padlock",
	"paintbrush",
	"pajama",
	"palace",
	"panpipe",
	"paper towel",
	"parachute",
	"parallel bars",
	"park bench",
	"parking meter",
	"passenger car",
	"patio",
	"pay-phone",
	"pedestal",
	"pencil box",
	"pencil sharpener",
	"perfume",
	"Petri dish",
	"photocopier",
	"pick",
	"pickelhaube",
	"picket fence",
	"pickup",
	"pier",
	"piggy bank",
	"pill bottle",
	"pillow",
	"ping-pong ball",
	"pinwheel",
	"pirate",
	"pitcher",
	"plane",
	"planetarium",
	"plastic bag",
	"plate rack",
	"plow",
	"plunger",
	"Polaroid camera",
	"pole",
	"police van",
	"poncho",
	"pool table",
	"pop bottle",
	"pot",
	"potter's wheel",
	"power drill",
	"prayer rug",
	"printer",
	"prison",
	"projectile",
	"projector",
	"puck",
	"punching bag",
	"purse",
	"quill",
	"quilt",
	"racer",
	"racket",
	"radiator",
	"radio",
	"radio telescope",
	"rain barrel",
	"recreational vehicle",
	"reel",
	"reflex camera",
	"refrigerator",
	"remote control",
	"restaurant",
	"revolver",
	"rifle",
	"rocking chair",
	"rotisserie",
	"rubber eraser",
	"rugby ball",
	"rule",
	"running shoe",
	"safe",
	"safety pin",
	"saltshaker",
	"sandal",
	"sarong",
	"sax",
	"scabbard",
	"scale",
	"school bus",
	"schooner",
	"scoreboard",
	"screen",
	"screw",
	"screwdriver",
	"seat belt",
	"sewing machine",
	"shield",
	"shoe shop",
	"shoji",
	"shopping basket",
	"shopping cart",
	"shovel",
	"shower cap",
	"shower curtain",
	"ski",
	"ski mask",
	"sleeping bag",
	"slide rule",
	"sliding door",
	"slot",
	"snorkel",
	"snowmobile",
	"snowplow",
	"soap dispenser",
	"soccer ball",
	"sock",
	"solar dish",
	"sombrero",
	"soup bowl",
	"space bar",
	"space heater",
	"space shuttle",
	"spatula",
	"speedboat",
	"spider web",
	"spindle",
	"sports car",
	"spotlight",
	"stage",
	"steam locomotive",
	"steel arch bridge",
	"steel drum",
	"stethoscope",
	"stole",
	"stone wall",
	"stopwatch",
	"stove",
	"strainer",
	"streetcar",
	"stretcher",
	"studio couch",
	"stupa",
	"submarine",
	"suit",
	"sundial",
	"sunglass",
	"sunglasses",
	"sunscreen",
	"suspension bridge",
	"swab",
	"sweatshirt",
	"swimming trunks",
	"swing",
	"switch",
	"syringe",
	"table lamp",
	"tank",
	"tape player",
	"teapot",
	"teddy",
	"television",
	"tennis ball",
	"thatch",
	"theater curtain",
	"thimble",
	"thresher",
	"throne",
	"tile roof",
	"toaster",
	"tobacco shop",
	"toilet seat",
	"torch",
	"totem pole",
	"tow truck",
	"toyshop",
	"tractor",
	"trailer truck",
	"tray",
	"trench coat",
	"tricycle",
	"trimaran",
	"tripod",
	"triumphal arch",
	"trolleybus",
	"trombone",
	"tub",
	"turnstile",
	"typewriter keyboard",
	"umbrella",
	"unicycle",
	"upright",
	"vacuum",
	"vase",
	"vault",
	"velvet",
	"vending machine",
	"vestment",
	"viaduct",
	"violin",
	"volleyball",
	"waffle iron",
	"wall clock",
	"wallet",
	"wardrobe",
	"warplane",
	"washbasin",
	"washer",
	"water bottle",
	"water jug",
	"water tower",
	"whiskey jug",
	"whistle",
	"wig",
	"window screen",
	"window shade",
	"Windsor tie",
	"wine bottle",
	"wing",
	"wok",
	"wooden spoon",
	"wool",
	"worm fence",
	"wreck",
	"yawl",
	"yurt",
	"web site",
	"comic book",
	"crossword puzzle",
	"street sign",
	"traffic light",
	"book jacket",
	"menu",
	"plate",
	"guacamole",
	"consomme",
	"hot pot",
	"trifle",
	"ice cream",
	"ice lolly",
	"French loaf",
	"bagel",
	"pretzel",
	"cheeseburger",
	"hotdog",
	"mashed potato",
	"head cabbage",
	"broccoli",
	"cauliflower",
	"zucchini",
	"spaghetti squash",
	"acorn squash",
	"butternut squash",
	"cucumber",
	"artichoke",
	"bell pepper",
	"cardoon",
	"mushroom",
	"Granny Smith",
	"strawberry",
	"orange",
	"lemon",
	"fig",
	"pineapple",
	"banana",
	"jackfruit",
	"custard apple",
	"pomegranate",
	"hay",
	"carbonara",
	"chocolate sauce",
	"dough",
	"meat loaf",
	"pizza",
	"potpie",
	"burrito",
	"red wine",
	"espresso",
	"cup",
	"eggnog",
	"alp",
	"bubble",
	"cliff",
	"coral reef",
	"geyser",
	"lakeside",
	"promontory",
	"sandbar",
	"seashore",
	"valley",
	"volcano",
	"ballplayer",
	"groom",
	"scuba diver",
	"rapeseed",
	"daisy",
	"yellow lady's slipper",
	"corn",
	"acorn",
	"hip",
	"buckeye",
	"coral fungus",
	"agaric",
	"gyromitra",
	"stinkhorn",
	"earthstar",
	"hen-of-the-woods",
	"bolete",
	"ear",
	"toilet tissue"
};
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

char *coco_classes[80] = {"person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"};

char *voc_classes[20] = { "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant","sheep","sofa", "train", "tv/monitor"};

fix16_t calcIou_LTRB(fix16_t* A, fix16_t* B){
    // pointers to elements (left, top, right, bottom)
    fix16_t left = MAX(A[0], B[0])>>4;  // adjust precision to prevent overflow; this is sufficient for 1920x1080
    fix16_t top = MAX(A[1], B[1])>>4;
    fix16_t right = MIN(A[2], B[2])>>4;
    fix16_t bottom = MIN(A[3], B[3])>>4;
    fix16_t i = fix16_mul(MAX(0,right-left), MAX(0,bottom-top));    // intersection
    fix16_t u = fix16_mul((A[2]-A[0])>>4, (A[3]-A[1])>>4) + fix16_mul((B[2]-B[0])>>4, (B[3]-B[1])>>4) - i;  // union
    if(u>0){
        fix16_t iou = fix16_div(i, u);
        return iou;
    }
    else{
        return 0;
    }
}
fix16_t calcIou_XYWH(fix16_t* A, fix16_t* B){
    // pointers to elements (x, y, width, height)
    fix16_t left = MAX(A[0] - (A[2]>>1), B[0] - (B[2]>>1));
    fix16_t right = MIN(A[0] + (A[2]>>1), B[0] + (B[2]>>1));
    fix16_t top = MAX(A[1] - (A[3]>>1), B[1] - (B[3]>>1));
    fix16_t bottom = MIN(A[1] + (A[3]>>1), B[1] + (B[3]>>1));
    fix16_t i = fix16_mul(MAX(0,right-left), MAX(0,bottom-top));    // intersection
    fix16_t u = fix16_mul(A[2], A[3]) + fix16_mul(B[2], B[3]) - i;  // union
    if(u>0)
        return fix16_div(i, u);
    else
        return 0;
}

static fix16_t fix16_eight = F16(8);
static fix16_t fix16_neight = F16(-8);
static fix16_t fix16_two = F16(2);
static fix16_t fix16_nhalf = F16(-0.5);


int partition(fix16_t* arr, int16_t *index, const int lo, const int hi)
{
	fix16_t temp, pivot = arr[hi];
	int itemp;
	int j, i = lo - 1;

	for (j = lo; j < hi; j++) {
		//since many elements are the same as pivot, "randomly" put them on either side of pivot
		//otherwise very slow
		int cmp = arr[j] == pivot ? j&1: arr[j] < pivot;
		if (cmp) {
			i++;
			temp = arr[j];
			arr[j] = arr[i];
			arr[i] = temp;

			itemp = index[j];
			index[j] = index[i];
			index[i] = itemp;
		}
	}

	if (arr[hi] < arr[i+1]) {
		temp = arr[hi];
		arr[hi] = arr[i+1];
		arr[i+1] = temp;

		itemp = index[hi];
		index[hi] = index[i+1];
		index[i+1] = itemp;
	}

	return i + 1;
}

void _quicksort(fix16_t *arr, int16_t *index, const int lo, const int hi)
{
	int split;
	if (lo < hi) {
		split = partition(arr, index, lo, hi);
		_quicksort(arr, index, lo, split-1);
		_quicksort(arr, index, split+1, hi);
	}
}

void quicksort(fix16_t *arr, int16_t *index, const int length)
{
	_quicksort(arr, index, 0, length-1);
}

void reverse(fix16_t* output_buffer[], int len){
	int left, right;
	for(left = 0, right = len-1; left < right; left++, right--){
		fix16_t* temp = output_buffer[left];
		output_buffer[left] = output_buffer[right];
		output_buffer[right] = temp;
	}
}

void post_classifier(fix16_t *outputs, const int out_sz, int16_t* output_index, int topk)
{
	int i;
	int16_t *idx = (int16_t*)malloc(out_sz*sizeof(int16_t));
	for (i = 0; i < out_sz; i++) idx[i] = i;
	quicksort(outputs, idx, out_sz);

	for(int i=0;i<topk;++i){
		output_index[i] = idx[out_sz-1 -i];
	}
	free(idx);
}


int close(float a, float b, float threshold) {
	float diff = a - b;
	if (diff < 0) diff = diff * -1;
	if (diff > threshold) return 0;
	return 1;
}

static fix16_t exp_lut[] ={0x00010000,0x00011082,0x00012216,0x000134cb,0x000148b5,0x00015de9,0x0001747a,0x00018c80,
	0x0001a612,0x0001c14b,0x0001de45,0x0001fd1d,0x00021df3,0x000240e7,0x0002661c,0x00028db8,
	0x0002b7e1,0x0002e4c2,0x00031489,0x00034764,0x00037d87,0x0003b727,0x0003f47f,0x000435cc,
	0x00047b4f,0x0004c54e,0x00051413,0x000567ec,0x0005c12d,0x00062030,0x00068554,0x0006f0fe,
	0x00076399,0x0007dd98,0x00085f76,0x0008e9b4,0x00097cdc,0x000a1982,0x000ac042,0x000b71c3,
	0x000c2eb7,0x000cf7db,0x000dcdf8,0x000eb1e4,0x000fa483,0x0010a6c8,0x0011b9b5,0x0012de5d,
	0x001415e5,0x00156185,0x0016c288,0x00183a4f,0x0019ca53,0x001b7423,0x001d396a,0x001f1bed,
	0x00211d8e,0x0023404f,0x00258654,0x0027f1e2,0x002a8565,0x002d4371,0x00302ec5,0x00334a4b,
	0x00369920,0x003a1e92,0x003dde28,0x0041dba1,0x00461afc,0x004aa077,0x004f7099,0x00549032,
	0x005a0462,0x005fd29e,0x006600b5,0x006c94d5,0x00739593,0x007b09f0,0x0082f962,0x008b6bd7,
	0x009469c4,0x009dfc28,0x00a82c94,0x00b3053b,0x00be90f6,0x00cadb53,0x00d7f09b,0x00e5dde6,
	0x00f4b122,0x01047924,0x011545b4,0x012727a1,0x013a30cf,0x014e7447,0x01640650,0x017afc7c,
	0x01936dc5,0x01ad729d,0x01c9250b,0x01e6a0c5,0x02060348,0x02276bf9,0x024afc45,0x0270d7bd,
	0x02992442,0x02c40a22,0x02f1b447,0x0322505f,0x03560f0b,0x038d240c,0x03c7c67e,0x04063107,
	0x0448a216,0x048f5c23,0x04daa5ee,0x052acac6,0x05801ad7,0x05daeb78,0x063b9782,0x06a27fa8,
	0x07100adb,0x0784a6b0,0x0800c7cc,0x0884ea5b,0x09119289,0x09a74d0c,0x0a46afaa,0x0af059d2,
	0x00000015,0x00000017,0x00000018,0x0000001a,0x0000001c,0x0000001e,0x0000001f,0x00000022,
	0x00000024,0x00000026,0x00000029,0x0000002b,0x0000002e,0x00000031,0x00000034,0x00000038,
	0x0000003b,0x0000003f,0x00000043,0x00000048,0x0000004c,0x00000051,0x00000056,0x0000005c,
	0x00000062,0x00000068,0x0000006f,0x00000076,0x0000007e,0x00000086,0x0000008f,0x00000098,
	0x000000a2,0x000000ac,0x000000b8,0x000000c3,0x000000d0,0x000000de,0x000000ec,0x000000fb,
	0x0000010b,0x0000011d,0x0000012f,0x00000143,0x00000157,0x0000016e,0x00000185,0x0000019e,
	0x000001b9,0x000001d6,0x000001f4,0x00000214,0x00000236,0x0000025b,0x00000282,0x000002ab,
	0x000002d8,0x00000306,0x00000338,0x0000036e,0x000003a6,0x000003e3,0x00000423,0x00000467,
	0x000004b0,0x000004fd,0x00000550,0x000005a7,0x00000605,0x00000668,0x000006d2,0x00000743,
	0x000007bb,0x0000083a,0x000008c2,0x00000953,0x000009ed,0x00000a90,0x00000b3f,0x00000bf9,
	0x00000cbe,0x00000d91,0x00000e71,0x00000f5f,0x0000105d,0x0000116b,0x0000128b,0x000013bd,
	0x00001503,0x0000165e,0x000017cf,0x00001958,0x00001afb,0x00001cb8,0x00001e93,0x0000208b,
	0x000022a5,0x000024e1,0x00002742,0x000029ca,0x00002c7c,0x00002f5a,0x00003268,0x000035a9,
	0x0000391f,0x00003cce,0x000040ba,0x000044e6,0x00004958,0x00004e13,0x0000531c,0x00005878,
	0x00005e2d,0x00006440,0x00006ab7,0x00007199,0x000078ed,0x000080b9,0x00008906,0x000091dd,
	0x00009b45,0x0000a549,0x0000aff2,0x0000bb4b,0x0000c75f,0x0000d43b,0x0000e1eb,0x0000f07d};

fix16_t fix16_exp_lut(fix16_t in){
	if (in < fix16_neight) return 0x15;
	if (in >= fix16_eight) return 0xaf059d0;

	uint8_t index = in >> 12;
	return exp_lut[index];
}

#define fix16_exp fix16_exp_lut
fix16_t fix16_logistic_activate(fix16_t x){ return fix16_div(fix16_one, fix16_add(fix16_one, fix16_exp(-x)));} // 1 div, 1 exp

void fix16_softmax(fix16_t *input, int n, fix16_t *output)
{
	fix16_t e, sum = 0;
	fix16_t largest = fix16_minimum;
	for(int i = 0; i < n; ++i){
		if(input[i] > largest) largest = input[i];
	}
	for(int i = 0; i < n; ++i){
		e = fix16_exp(fix16_sub(input[i], largest));
		sum = fix16_add(sum, e);
		output[i] = e;
	}
	fix16_t inv_sum = fix16_div(fix16_one, sum);
	for(int i = 0; i < n; ++i){
		output[i] = fix16_mul(output[i], inv_sum);
	}
}

int fix16_box_iou(fix16_box box_1, fix16_box box_2, fix16_t thresh)
{
	//return true if the IOU score of box_1 and box2 > threshold
	int width_of_overlap_area = ((box_1.xmax < box_2.xmax) ? box_1.xmax : box_2.xmax) - ((box_1.xmin > box_2.xmin) ? box_1.xmin : box_2.xmin);
	int height_of_overlap_area = ((box_1.ymax < box_2.ymax) ? box_1.ymax : box_2.ymax) - ((box_1.ymin > box_2.ymin) ? box_1.ymin : box_2.ymin);

	int area_of_overlap;
	if (width_of_overlap_area < 0 || height_of_overlap_area < 0) {
		return 0;
	} else {
		area_of_overlap = width_of_overlap_area * height_of_overlap_area;
		if( area_of_overlap <0){
			//overflow,divide boxes by 4 and try again
			box_1.xmin >>=2;
			box_1.ymin >>=2;
			box_1.xmax >>=2;
			box_1.ymax >>=2;
			box_2.xmin >>=2;
			box_2.ymin >>=2;
			box_2.xmax >>=2;
			box_2.ymax >>=2;
			return fix16_box_iou(box_1,box_2,thresh);
		}
	}

	int box_1_area = (box_1.ymax - box_1.ymin) * (box_1.xmax - box_1.xmin);
	int box_2_area = (box_2.ymax - box_2.ymin) * (box_2.xmax - box_2.xmin);
	int area_of_union = box_1_area + box_2_area - area_of_overlap;
	if (area_of_union == 0) return 0;

	return area_of_overlap > fix16_mul(thresh, area_of_union);
}




void fix16_do_nms(fix16_box *boxes, int total, fix16_t iou_thresh)
{
	for(int i = 0; i < total; i++){
		if (boxes[i].confidence == 0) continue;
		for(int j = i+1; j < total; j++){
			if (fix16_box_iou(boxes[i], boxes[j], iou_thresh)){
				boxes[j].confidence = 0;
			}
		}
	}
}

void fix16_sort_boxes(fix16_box *boxes, int total)
{
	for (int i = 0; i < total-1; i++) {
		int max_id = i;
		for(int j = i+1; j < total; j++){
			if (boxes[max_id].confidence < boxes[j].confidence) {
				max_id = j;
			}
		}

		if (max_id != i) {
			//swap max_id with i
			fix16_box tmp = boxes[i];
			memcpy((void*)(boxes+i), (void*)(boxes+max_id), sizeof(fix16_box));
			memcpy((void*)(boxes+max_id), &tmp, sizeof(fix16_box));
		}
	}
}


int fix16_clean_boxes(fix16_box *boxes, int total, int width, int height)
{
	int b=0;
	for(int i = 0; i < total; i++){
		if (boxes[i].confidence > 0) {
			fix16_box box = boxes[i];
			box.xmin = box.xmin < 0 ? 0 : box.xmin;
			box.xmax = box.xmax >width ? width : box.xmax;
			box.ymin = box.ymin < 0 ? 0 : box.ymin;
			box.ymax = box.ymax >height ? height : box.ymax;
			boxes[b++] = box;
		}
	}
	return b;
}




int fix16_get_region_boxes(fix16_t *predictions, fix16_t *biases, const int w, const int h, const int ln, const int classes, fix16_t w_ratio, fix16_t h_ratio, fix16_t thresh, fix16_t log_odds, fix16_box *boxes,int max_boxes, const int do_logistic, const int do_softmax, const int version)
{
	int box_count = 0;

	int num_size=(classes+5) *w*h;
	fix16_t box[classes+4];
	for(int r=0;r<h;r++){
		for(int c=0;c<w;c++){
			fix16_t row = fix16_from_int(r);
			fix16_t col = fix16_from_int(c);
			for(int n = 0; n < ln; ++n){
				int p_index = n*num_size+4*w*h+r*w+c;
				fix16_t scale = predictions[p_index];
				if (do_logistic) {
					if (scale < log_odds) continue;
					scale = fix16_logistic_activate(scale);
				}
				if (scale < thresh) continue;

				const int class_offset = 4;
				for(int j=0; j < class_offset;++j){
					box[j] = predictions[n*num_size+j*w*h+r*w+c];
				}
				for(int j=0;j<classes;++j){
					box[j+class_offset] = predictions[n*num_size+(j+class_offset+1)*w*h+r*w+c];
				}

				fix16_t bx, by, bw, bh;
				if (version > 3) {
					// (col+logisitic(box)*2-0.5) * ratio
					bx = fix16_mul(fix16_add(fix16_add(col, fix16_mul(fix16_logistic_activate(box[0]), fix16_two)), fix16_nhalf), w_ratio);
					by = fix16_mul(fix16_add(fix16_add(row, fix16_mul(fix16_logistic_activate(box[1]), fix16_two)), fix16_nhalf), h_ratio);

					bh = fix16_mul(fix16_logistic_activate(box[3]), fix16_two);
					bh = fix16_mul(fix16_mul(bh, bh), biases[2*n+1]);

					// (logisitic(box)*2)**2 * anchor
					bw = fix16_mul(fix16_logistic_activate(box[2]), fix16_two);
					bw = fix16_mul(fix16_mul(bw, bw), biases[2*n]);
				} else {
					bx = fix16_mul(fix16_add(col, fix16_logistic_activate(box[ 0])), w_ratio);
					by = fix16_mul(fix16_add(row, fix16_logistic_activate(box[ 1])), h_ratio);
					bw = fix16_mul(fix16_exp(box[2]), biases[2*n]);
					bh = fix16_mul(fix16_exp(box[3]), biases[2*n+1]);
				}

				if (do_softmax) {
					if (version < 3) {
						fix16_softmax(box + class_offset, classes, box + class_offset);
					} else {
						for(int j=0;j<classes;++j){
							box[j+class_offset] = fix16_logistic_activate(box[j+class_offset]);
						}
					}
				}

				for(int j = 0; j < classes; j++){
					fix16_t prob = fix16_mul(scale, box[class_offset+j]);
					if (prob > thresh) {
						fix16_t xmin = fix16_sub(bx, bw >> 1);
						fix16_t ymin = fix16_sub(by, bh >> 1);
						fix16_t xmax = fix16_add(xmin, bw);
						fix16_t ymax = fix16_add(ymin, bh);

						boxes[box_count].xmin = fix16_to_int(xmin);
						boxes[box_count].ymin = fix16_to_int(ymin);
						boxes[box_count].xmax = fix16_to_int(xmax);
						boxes[box_count].ymax = fix16_to_int(ymax);
						boxes[box_count].confidence = prob;
						boxes[box_count].class_id = j;
						box_count++;
						if(box_count == max_boxes){
							return box_count;
						}
					}
				}
			}
		}
	}
	return box_count;
}

int post_process_yolo(fix16_t **outputs, const int num_outputs, yolo_info_t *cfg,
		fix16_t thresh, fix16_t overlap, fix16_box fix16_boxes[], int max_boxes)
{
	int total_box_count = 0;
	int input_h  = cfg[0].input_dims[1];
	int input_w  = cfg[0].input_dims[2];
	fix16_t *anchors = cfg[0].anchors;

	fix16_t fix16_log_odds = fix16_log(fix16_div(thresh, fix16_sub(fix16_one, thresh)));
	for (int o = 0; o < num_outputs; o++) {
		fix16_t *out32 = outputs[o];

		int num_per_output = cfg[o].num;
		if (num_outputs > 1) num_per_output = cfg[o].mask_length;


		fix16_t fix16_biases[2*num_per_output];

		int h  = cfg[o].output_dims[1];
		int w  = cfg[o].output_dims[2];
		fix16_t h_ratio = fix16_from_int(input_h/h);
		fix16_t w_ratio = fix16_from_int(input_w/w);

		for (int i = 0; i < num_per_output; i++) {
			int mask = i;
			if (num_outputs > 1) mask = cfg[o].mask[i];

			if (cfg[o].version == 2) {
				fix16_biases[2*i] = fix16_mul(anchors[2*mask], w_ratio);
				fix16_biases[2*i+1] = fix16_mul(anchors[2*mask+1], h_ratio);
			} else {
				fix16_biases[2*i] = anchors[2*mask];
				fix16_biases[2*i+1] = anchors[2*mask+1];
			}
		}

		int fix16_box_count = fix16_get_region_boxes(out32, fix16_biases, w, h, num_per_output, cfg[o].classes, w_ratio,
				h_ratio, thresh, fix16_log_odds,
				fix16_boxes + total_box_count,max_boxes-total_box_count, 1, 1, cfg[o].version);
		fflush(stdout);

		// copy boxes
		total_box_count += fix16_box_count;
		if(total_box_count == max_boxes){
			break;
		}
	}

	fix16_sort_boxes(fix16_boxes, total_box_count);
	fix16_do_nms(fix16_boxes, total_box_count, overlap);

	int clean_box_count = fix16_clean_boxes(fix16_boxes, total_box_count, input_w, input_h);

	return clean_box_count;
}

void post_process_classifier(fix16_t *outputs, const int out_sz, int16_t* output_index, int topk)
{

	fix16_t* cached_output=malloc(out_sz*sizeof(*cached_output));
	for(int i=0;i<out_sz;++i){
		cached_output[i] = outputs[i];
	}
	post_classifier(cached_output, out_sz, output_index, topk);
	free(cached_output);
}


void ctc_raw_indices(int *indices, fix16_t *output, const int output_len, const int output_depth)
{
    for (int x = 0; x < output_len; x++){
	    int max_idx = 0;
	    fix16_t max = -1;

	    for (int y = 0; y < output_depth; y++){
		    if (output[y*output_len + x] > max) {
			    max = output[y*output_len + x];
			    max_idx = y;
		    }
	    }

	    if (max > 0) {
		    indices[x] = max_idx;
	    } else {
		    indices[x] = -1;
	    }
    }
}


void ctc_greedy_decode(int *unique, fix16_t *output, const int output_len, const int output_depth, int merge_repeated)
{
    int blank_index = output_depth - 1;
    int indices[output_len]; 
    ctc_raw_indices(indices, output, output_len, output_depth);

    int idx, prev = -1, uidx = 0;
    for (int i = 0; i < output_len; i++){
	    unique[i] = -1;
	    idx = indices[i];
	    if (idx != blank_index && idx != -1) {
		    if(merge_repeated) { 
			if (prev == -1 || prev != idx) {
			    prev = idx;
			    unique[uidx]=idx;
			    uidx++;
			}
		    } else {
			    unique[uidx]=idx;
			    uidx++;
		    }
		    
	    } else {
		    prev = -1;
	    }
    }
}


#ifndef MIN
#  define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#  define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif


static const unsigned char blazeAnchors256[896][2] =
{{8,8},{8,8},{24,8},{24,8},{40,8},{40,8},{56,8},{56,8},
	{72,8},{72,8},{88,8},{88,8},{104,8},{104,8},{120,8},{120,8},{136,8},{136,8},{152,8},{152,8},{168,8},{168,8},{184,8},{184,8},
	{200,8},{200,8},{216,8},{216,8},{232,8},{232,8},{248,8},{248,8},{8,24},{8,24},{24,24},{24,24},{40,24},{40,24},{56,24},{56,24},
	{72,24},{72,24},{88,24},{88,24},{104,24},{104,24},{120,24},{120,24},{136,24},{136,24},{152,24},{152,24},{168,24},{168,24},{184,24},{184,24},
	{200,24},{200,24},{216,24},{216,24},{232,24},{232,24},{248,24},{248,24},{8,40},{8,40},{24,40},{24,40},{40,40},{40,40},{56,40},{56,40},
	{72,40},{72,40},{88,40},{88,40},{104,40},{104,40},{120,40},{120,40},{136,40},{136,40},{152,40},{152,40},{168,40},{168,40},{184,40},{184,40},
	{200,40},{200,40},{216,40},{216,40},{232,40},{232,40},{248,40},{248,40},{8,56},{8,56},{24,56},{24,56},{40,56},{40,56},{56,56},{56,56},
	{72,56},{72,56},{88,56},{88,56},{104,56},{104,56},{120,56},{120,56},{136,56},{136,56},{152,56},{152,56},{168,56},{168,56},{184,56},{184,56},
	{200,56},{200,56},{216,56},{216,56},{232,56},{232,56},{248,56},{248,56},{8,72},{8,72},{24,72},{24,72},{40,72},{40,72},{56,72},{56,72},
	{72,72},{72,72},{88,72},{88,72},{104,72},{104,72},{120,72},{120,72},{136,72},{136,72},{152,72},{152,72},{168,72},{168,72},{184,72},{184,72},
	{200,72},{200,72},{216,72},{216,72},{232,72},{232,72},{248,72},{248,72},{8,88},{8,88},{24,88},{24,88},{40,88},{40,88},{56,88},{56,88},
	{72,88},{72,88},{88,88},{88,88},{104,88},{104,88},{120,88},{120,88},{136,88},{136,88},{152,88},{152,88},{168,88},{168,88},{184,88},{184,88},
	{200,88},{200,88},{216,88},{216,88},{232,88},{232,88},{248,88},{248,88},{8,104},{8,104},{24,104},{24,104},{40,104},{40,104},{56,104},{56,104},
	{72,104},{72,104},{88,104},{88,104},{104,104},{104,104},{120,104},{120,104},{136,104},{136,104},{152,104},{152,104},{168,104},{168,104},{184,104},{184,104},
	{200,104},{200,104},{216,104},{216,104},{232,104},{232,104},{248,104},{248,104},{8,120},{8,120},{24,120},{24,120},{40,120},{40,120},{56,120},{56,120},
	{72,120},{72,120},{88,120},{88,120},{104,120},{104,120},{120,120},{120,120},{136,120},{136,120},{152,120},{152,120},{168,120},{168,120},{184,120},{184,120},
	{200,120},{200,120},{216,120},{216,120},{232,120},{232,120},{248,120},{248,120},{8,136},{8,136},{24,136},{24,136},{40,136},{40,136},{56,136},{56,136},
	{72,136},{72,136},{88,136},{88,136},{104,136},{104,136},{120,136},{120,136},{136,136},{136,136},{152,136},{152,136},{168,136},{168,136},{184,136},{184,136},
	{200,136},{200,136},{216,136},{216,136},{232,136},{232,136},{248,136},{248,136},{8,152},{8,152},{24,152},{24,152},{40,152},{40,152},{56,152},{56,152},
	{72,152},{72,152},{88,152},{88,152},{104,152},{104,152},{120,152},{120,152},{136,152},{136,152},{152,152},{152,152},{168,152},{168,152},{184,152},{184,152},
	{200,152},{200,152},{216,152},{216,152},{232,152},{232,152},{248,152},{248,152},{8,168},{8,168},{24,168},{24,168},{40,168},{40,168},{56,168},{56,168},
	{72,168},{72,168},{88,168},{88,168},{104,168},{104,168},{120,168},{120,168},{136,168},{136,168},{152,168},{152,168},{168,168},{168,168},{184,168},{184,168},
	{200,168},{200,168},{216,168},{216,168},{232,168},{232,168},{248,168},{248,168},{8,184},{8,184},{24,184},{24,184},{40,184},{40,184},{56,184},{56,184},
	{72,184},{72,184},{88,184},{88,184},{104,184},{104,184},{120,184},{120,184},{136,184},{136,184},{152,184},{152,184},{168,184},{168,184},{184,184},{184,184},
	{200,184},{200,184},{216,184},{216,184},{232,184},{232,184},{248,184},{248,184},{8,200},{8,200},{24,200},{24,200},{40,200},{40,200},{56,200},{56,200},
	{72,200},{72,200},{88,200},{88,200},{104,200},{104,200},{120,200},{120,200},{136,200},{136,200},{152,200},{152,200},{168,200},{168,200},{184,200},{184,200},
	{200,200},{200,200},{216,200},{216,200},{232,200},{232,200},{248,200},{248,200},{8,216},{8,216},{24,216},{24,216},{40,216},{40,216},{56,216},{56,216},
	{72,216},{72,216},{88,216},{88,216},{104,216},{104,216},{120,216},{120,216},{136,216},{136,216},{152,216},{152,216},{168,216},{168,216},{184,216},{184,216},
	{200,216},{200,216},{216,216},{216,216},{232,216},{232,216},{248,216},{248,216},{8,232},{8,232},{24,232},{24,232},{40,232},{40,232},{56,232},{56,232},
	{72,232},{72,232},{88,232},{88,232},{104,232},{104,232},{120,232},{120,232},{136,232},{136,232},{152,232},{152,232},{168,232},{168,232},{184,232},{184,232},
	{200,232},{200,232},{216,232},{216,232},{232,232},{232,232},{248,232},{248,232},{8,248},{8,248},{24,248},{24,248},{40,248},{40,248},{56,248},{56,248},
	{72,248},{72,248},{88,248},{88,248},{104,248},{104,248},{120,248},{120,248},{136,248},{136,248},{152,248},{152,248},{168,248},{168,248},{184,248},{184,248},
	{200,248},{200,248},{216,248},{216,248},{232,248},{232,248},{248,248},{248,248},{16,16},{16,16},{16,16},{16,16},{16,16},{16,16},{48,16},{48,16},
	{48,16},{48,16},{48,16},{48,16},{80,16},{80,16},{80,16},{80,16},{80,16},{80,16},{112,16},{112,16},{112,16},{112,16},{112,16},{112,16},
	{144,16},{144,16},{144,16},{144,16},{144,16},{144,16},{176,16},{176,16},{176,16},{176,16},{176,16},{176,16},{208,16},{208,16},{208,16},{208,16},
	{208,16},{208,16},{240,16},{240,16},{240,16},{240,16},{240,16},{240,16},{16,48},{16,48},{16,48},{16,48},{16,48},{16,48},{48,48},{48,48},
	{48,48},{48,48},{48,48},{48,48},{80,48},{80,48},{80,48},{80,48},{80,48},{80,48},{112,48},{112,48},{112,48},{112,48},{112,48},{112,48},
	{144,48},{144,48},{144,48},{144,48},{144,48},{144,48},{176,48},{176,48},{176,48},{176,48},{176,48},{176,48},{208,48},{208,48},{208,48},{208,48},
	{208,48},{208,48},{240,48},{240,48},{240,48},{240,48},{240,48},{240,48},{16,80},{16,80},{16,80},{16,80},{16,80},{16,80},{48,80},{48,80},
	{48,80},{48,80},{48,80},{48,80},{80,80},{80,80},{80,80},{80,80},{80,80},{80,80},{112,80},{112,80},{112,80},{112,80},{112,80},{112,80},
	{144,80},{144,80},{144,80},{144,80},{144,80},{144,80},{176,80},{176,80},{176,80},{176,80},{176,80},{176,80},{208,80},{208,80},{208,80},{208,80},
	{208,80},{208,80},{240,80},{240,80},{240,80},{240,80},{240,80},{240,80},{16,112},{16,112},{16,112},{16,112},{16,112},{16,112},{48,112},{48,112},
	{48,112},{48,112},{48,112},{48,112},{80,112},{80,112},{80,112},{80,112},{80,112},{80,112},{112,112},{112,112},{112,112},{112,112},{112,112},{112,112},
	{144,112},{144,112},{144,112},{144,112},{144,112},{144,112},{176,112},{176,112},{176,112},{176,112},{176,112},{176,112},{208,112},{208,112},{208,112},{208,112},
	{208,112},{208,112},{240,112},{240,112},{240,112},{240,112},{240,112},{240,112},{16,144},{16,144},{16,144},{16,144},{16,144},{16,144},{48,144},{48,144},
	{48,144},{48,144},{48,144},{48,144},{80,144},{80,144},{80,144},{80,144},{80,144},{80,144},{112,144},{112,144},{112,144},{112,144},{112,144},{112,144},
	{144,144},{144,144},{144,144},{144,144},{144,144},{144,144},{176,144},{176,144},{176,144},{176,144},{176,144},{176,144},{208,144},{208,144},{208,144},{208,144},
	{208,144},{208,144},{240,144},{240,144},{240,144},{240,144},{240,144},{240,144},{16,176},{16,176},{16,176},{16,176},{16,176},{16,176},{48,176},{48,176},
	{48,176},{48,176},{48,176},{48,176},{80,176},{80,176},{80,176},{80,176},{80,176},{80,176},{112,176},{112,176},{112,176},{112,176},{112,176},{112,176},
	{144,176},{144,176},{144,176},{144,176},{144,176},{144,176},{176,176},{176,176},{176,176},{176,176},{176,176},{176,176},{208,176},{208,176},{208,176},{208,176},
	{208,176},{208,176},{240,176},{240,176},{240,176},{240,176},{240,176},{240,176},{16,208},{16,208},{16,208},{16,208},{16,208},{16,208},{48,208},{48,208},
	{48,208},{48,208},{48,208},{48,208},{80,208},{80,208},{80,208},{80,208},{80,208},{80,208},{112,208},{112,208},{112,208},{112,208},{112,208},{112,208},
	{144,208},{144,208},{144,208},{144,208},{144,208},{144,208},{176,208},{176,208},{176,208},{176,208},{176,208},{176,208},{208,208},{208,208},{208,208},{208,208},
	{208,208},{208,208},{240,208},{240,208},{240,208},{240,208},{240,208},{240,208},{16,240},{16,240},{16,240},{16,240},{16,240},{16,240},{48,240},{48,240},
	{48,240},{48,240},{48,240},{48,240},{80,240},{80,240},{80,240},{80,240},{80,240},{80,240},{112,240},{112,240},{112,240},{112,240},{112,240},{112,240},
	{144,240},{144,240},{144,240},{144,240},{144,240},{144,240},{176,240},{176,240},{176,240},{176,240},{176,240},{176,240},{208,240},{208,240},{208,240},{208,240},
	{208,240},{208,240},{240,240},{240,240},{240,240},{240,240},{240,240},{240,240}};


static fix16_t calcIou(fix16_t* A, fix16_t* B){
	// pointers to elements (x, y, width, height)
	fix16_t left = MAX(A[0] - (A[2]>>1), B[0] - (B[2]>>1));
	fix16_t right = MIN(A[0] + (A[2]>>1), B[0] + (B[2]>>1));
	fix16_t top = MAX(A[1] - (A[3]>>1), B[1] - (B[3]>>1));
	fix16_t bottom = MIN(A[1] + (A[3]>>1), B[1] + (B[3]>>1));
	fix16_t i = fix16_mul(MAX(0,right-left), MAX(0,bottom-top));    // intersection
	fix16_t u = fix16_mul(A[2], A[3]) + fix16_mul(B[2], B[3]) - i;  // union
	if(u>0)
		return fix16_div(i, u);
	else
		return 0;
}

int post_process_blazeface(object_t faces[],fix16_t* scores,fix16_t* points,int scoresLength,int max_faces, fix16_t anchorsScale) {

	fix16_t min_suppression_threshold = F16(0.3);
	fix16_t raw_thresh = F16(1.0986122886681098);    // 1.0986122886681098 == -log((1-thresh)/thresh),  thresh=0.75

	int num_detects = 0;   // number of detects (before combining by blending)
	const int maxRawDetects = 64;
	int ind[maxRawDetects];    // indices of scores in descending order

	//int maxRawScores = 0;
	// find detections based on score and add to a sorted list of indices (indices of highest scores first)
	for(int n=0; n<scoresLength; n++){
		//maxRawScores = MAX(maxRawScores, scores[n]);
		if(scores[n] > raw_thresh){
			scores[n] = fix16_div(fix16_one, fix16_one + fix16_exp(-scores[n]));    // compute score
			int i=0;
			while(i<num_detects){ // find the insertion index
				if(scores[n] > scores[ind[i]]){
					for(int i2=MIN(num_detects,maxRawDetects-1); i2>i; i2--) // move down all lower elements
						ind[i2] = ind[i2-1];
					ind[i] = n;
					num_detects++;
					break;
				}
				i++;
			}
			if(i==num_detects && num_detects<maxRawDetects){   // if not inserted and there's room, then insert at the end
				ind[i] = n;
				num_detects++;
			}
		}
	}

	//tfp_printf("MAX RAW SCORE: %d \n", maxRawScores);

	// add achors to points (omitting indices 2 and 3, which are the width and height of the box)
	for(int i=0; i<num_detects; i++){
		int n = ind[i];
		fix16_t xAnchor = fix16_mul(fix16_from_int(blazeAnchors256[n][0]),anchorsScale);
		fix16_t yAnchor = fix16_mul(fix16_from_int(blazeAnchors256[n][1]),anchorsScale);
		points[n*16 + 0] += xAnchor;
		points[n*16 + 1] += yAnchor;
		for(int p=4; p<16; p+=2){
			points[n*16 + p] += xAnchor;
			points[n*16 + p+1] += yAnchor;
		}
	}

	// find overlapping detects and blend them together, then add to array of face structures
	int facesLength = 0;
	char used[num_detects];   // true if raw detect was blended with another detect (make sure it doesn't get used again)
	for(int i=0; i<num_detects; i++)
		used[i] = 0;
	for(int i1=0; i1<num_detects; i1++){
		if(used[i1])
			continue;
		fix16_t totalScore = scores[ind[i1]];
		fix16_t blendPoints[16];
		for(int p=0; p<16; p++)
			blendPoints[p] = fix16_mul(points[ind[i1]*16+p], scores[ind[i1]]);  // weight based on score
		for(int i2=i1+1; i2<num_detects; i2++){
			if(used[i2])
				continue;
			fix16_t iou = calcIou(&points[ind[i1]*16], &points[ind[i2]*16]);
			if(iou > min_suppression_threshold){
				used[i2] = 1;
				for(int p=0; p<16; p++)
					blendPoints[p] += fix16_mul(scores[ind[i2]], points[ind[i2]*16+p]);
				totalScore += scores[ind[i2]];
			}
		}
		fix16_t scale = fix16_div(fix16_one, totalScore);
		for(int p=0; p<16; p++)
			blendPoints[p] = fix16_mul(blendPoints[p], scale);   // scale back to original

		fix16_t x = blendPoints[0];
		fix16_t y = blendPoints[1];
		fix16_t w = blendPoints[2];
		fix16_t h = blendPoints[3];
		blendPoints[0] = x-w/2;   // convert from (x,y,w,h) to (left,top,right,bottom)
		blendPoints[1] = y-h/2;
		blendPoints[2] = x+w/2;
		blendPoints[3] = y+h/2;
		// write to class face struct
		for(int p=0; p<4; p++)
			faces[facesLength].box[p] = blendPoints[p];
		for(int p=0; p<4; p++){
			faces[facesLength].points[p][0] = blendPoints[p*2+4];
			faces[facesLength].points[p][1] = blendPoints[p*2+5];
		}
		faces[facesLength].detect_score = scores[ind[i1]];
		facesLength++;
		if(facesLength==max_faces)
			break;
	}
	return facesLength;
}

int pprint_post_process(const char *name, const char *str, model_t *model, vbx_cnn_io_ptr_t *io_buffers)
{
	
	if (!strcmp(str, "BLAZEFACE")){
		const int MAX_FACES=24;
		object_t faces[MAX_FACES];
		// reverse
		fix16_t* output_buffer0=(fix16_t*)(uintptr_t)io_buffers[2];
		fix16_t* output_buffer1=(fix16_t*)(uintptr_t)io_buffers[1];
		int output_length0 = model_get_output_length(model, 1);
		int output_length1 = model_get_output_length(model, 0);

		int facesLength = 0;
		if (output_length0 < output_length1) {
			facesLength = post_process_blazeface(faces,output_buffer0,output_buffer1,output_length0,
					MAX_FACES,fix16_from_int(1));
		} else {
			facesLength = post_process_blazeface(faces,output_buffer1,output_buffer0,output_length1,
					MAX_FACES,fix16_from_int(1));
		}
		for(int f=0;f<facesLength;f++){
			object_t* face = faces+f;
			fix16_t x = face->box[0];
			fix16_t y = face->box[1];
			fix16_t w = face->box[2] - face->box[0];
			fix16_t h = face->box[3] - face->box[1];
			printf("face %d found at (x,y,w,h) %3.1f %3.1f %3.1f %3.1f\n",f,
					fix16_to_float(x), fix16_to_float(y),
					fix16_to_float(w), fix16_to_float(h));
		}
	} else if (!strcmp(str,"RETINAFACE")){
		const int MAX_FACES=24;
		object_t faces[MAX_FACES];
		fix16_t confidence_threshold=F16(0.8);
		fix16_t nms_threshold=F16(0.4);

		fix16_t* output_buffers[9];
		//( 0 1 2 3 4 5 6 7 8)->(5 4 3 8 7 6 2 1 0)
		output_buffers[0]=(fix16_t*)(uintptr_t)io_buffers[1+5];
		output_buffers[1]=(fix16_t*)(uintptr_t)io_buffers[1+4];
		output_buffers[2]=(fix16_t*)(uintptr_t)io_buffers[1+3];
		output_buffers[3]=(fix16_t*)(uintptr_t)io_buffers[1+8];
		output_buffers[4]=(fix16_t*)(uintptr_t)io_buffers[1+7];
		output_buffers[5]=(fix16_t*)(uintptr_t)io_buffers[1+6];
		output_buffers[6]=(fix16_t*)(uintptr_t)io_buffers[1+2];
		output_buffers[7]=(fix16_t*)(uintptr_t)io_buffers[1+1];
		output_buffers[8]=(fix16_t*)(uintptr_t)io_buffers[1+0];

		int input_length = model_get_input_length(model,0);
		int image_h = 288;
		int image_w = 512;
		if (input_length == (3*320*320)) {
			image_h = 320;
			image_w = 320;
		} else if (input_length == (3*640*640)) {
			image_h = 640;
			image_w = 640;
		}

		int facesLength = post_process_retinaface(faces,MAX_FACES,output_buffers, image_w, image_h,
				confidence_threshold,nms_threshold);

		for(int f=0;f<facesLength;f++){
			object_t* face = faces+f;
			fix16_t x = face->box[0];
			fix16_t y = face->box[1];
			fix16_t w = face->box[2] - face->box[0];
			fix16_t h = face->box[3] - face->box[1];
			printf("face %d found at (x,y,w,h) %3.1f %3.1f %3.1f %3.1f\n",f,
					fix16_to_float(x), fix16_to_float(y),
					fix16_to_float(w), fix16_to_float(h));
			printf("landmarks: ");
			for(int l =0;l<5;++l){
				printf("%3.1f,%3.1f ",
						fix16_to_float(face->points[l][0]),
						fix16_to_float(face->points[l][1]));
				fflush(stdout);
			}printf("\n");


		}
	} else if (!strcmp(str,"LPD")) {
		const int MAX_PLATES=10;
		object_t plates[MAX_PLATES];
		fix16_t confidence_threshold=F16(0.55);
		fix16_t nms_threshold=F16(0.2);

		int num_outputs = model_get_num_outputs(model);
		int image_h = 288;
		int image_w = 1024;
	
		fix16_t* output_buffers[num_outputs];
		for (int o = 0; o < num_outputs; o++) {
			output_buffers[o]=(fix16_t*)(uintptr_t)io_buffers[1+o];
		}
		int platesLength = post_process_lpd(plates, MAX_PLATES, output_buffers, image_w, image_h,
				confidence_threshold,nms_threshold, num_outputs);

		for(int f=0;f<platesLength;f++){
			object_t* plate = plates+f;
			fix16_t x = plate->box[0];
			fix16_t y = plate->box[1];
			fix16_t w = plate->box[2];
			fix16_t h = plate->box[3];
			printf("plate %d found at (x,y,w,h) %3.1f %3.1f %3.1f %3.1f\n",f,
					fix16_to_float(x), fix16_to_float(y),
					fix16_to_float(w), fix16_to_float(h));
			printf("landmarks: ");
			for(int l =0;l<5;++l){
				printf("%3.1f,%3.1f ",
						fix16_to_float(plate->points[l][0]),
						fix16_to_float(plate->points[l][1]));
				fflush(stdout);
			}printf("\n");


		}
	} else if (!strcmp(str, "LPR")){
		char label[20];
		fix16_t* output=(fix16_t*)(uintptr_t)io_buffers[1];
		fix16_t conf = post_process_lpr(output, model_get_output_length(model, 0), label);
		printf("Plate ID: %s Recognition Score: %3.4f\n", label, fix16_to_float(conf));

	} else if (!strcmp(str,"SCRFD")) {
		const int MAX_FACES=24;
		object_t faces[MAX_FACES];
		fix16_t confidence_threshold=F16(0.8);
		fix16_t nms_threshold=F16(0.4);

		fix16_t* output_buffers[9];
		//( 0 1 2 3 4 5 6 7 8)->(2 5 8 1 4 7 0 3 6)
		output_buffers[0]=(fix16_t*)(uintptr_t)io_buffers[1+2];
		output_buffers[1]=(fix16_t*)(uintptr_t)io_buffers[1+5];
		output_buffers[2]=(fix16_t*)(uintptr_t)io_buffers[1+8];
		output_buffers[3]=(fix16_t*)(uintptr_t)io_buffers[1+1];
		output_buffers[4]=(fix16_t*)(uintptr_t)io_buffers[1+4];
		output_buffers[5]=(fix16_t*)(uintptr_t)io_buffers[1+7];
		output_buffers[6]=(fix16_t*)(uintptr_t)io_buffers[1+0];
		output_buffers[7]=(fix16_t*)(uintptr_t)io_buffers[1+3];
		output_buffers[8]=(fix16_t*)(uintptr_t)io_buffers[1+6];
		

		int input_length = model_get_input_length(model,0);
		int image_h = 288;
		int image_w = 512;
		if (input_length == (3*320*320)) {
			image_h = 320;
			image_w = 320;
		} else if (input_length == (3*640*640)) {
			image_h = 640;
			image_w = 640;
		}
	
		int facesLength = post_process_scrfd(faces,MAX_FACES,output_buffers, image_w, image_h,
				confidence_threshold,nms_threshold);

		for(int f=0;f<facesLength;f++){
			object_t* face = faces+f;
			fix16_t x = face->box[0];
			fix16_t y = face->box[1];
			fix16_t w = face->box[2] - face->box[0];
			fix16_t h = face->box[3] - face->box[1];
			printf("face %d found at (x,y,w,h) %3.1f %3.1f %3.1f %3.1f\n",f,
					fix16_to_float(x), fix16_to_float(y),
					fix16_to_float(w), fix16_to_float(h));
			printf("landmarks: ");
			for(int l =0;l<5;++l){
				printf("%3.1f,%3.1f ",
						fix16_to_float(face->points[l][0]),
						fix16_to_float(face->points[l][1]));
				fflush(stdout);
			}printf("\n");


		}
	} else if (!strcmp(str, "CLASSIFY")){
		const int topk=10;
		int16_t indices[topk];
		int output_length = model_get_output_length(model, 0);
		fix16_t* output_buffer0=(fix16_t*)(uintptr_t)io_buffers[1];
		post_process_classifier(output_buffer0,output_length,indices,topk);
		for(int i = 0;i < topk; ++i){
			int idx = indices[i];
			int score = output_buffer0[idx];
			if(output_length == 1001 || output_length == 1000){ // imagenet
				char* class_name = imagenet_classes[idx];
				if(output_length==1001){
					//some imagenet networks have a null catagory, account for that
					class_name =  imagenet_classes[idx-1];
				}
				printf("%d: %d - %s = %3.2f\n", i, idx, class_name, fix16_to_float(score));
			} else {
				printf("%d: %d = %3.2f\n", i, idx, fix16_to_float(score));
			}
		}
	} else if (!strcmp(str, "PLATE")){
		fix16_t* output=(fix16_t*)(uintptr_t)io_buffers[1];
		char **lpr = NULL;
		int output_len, output_depth;
		int merge_repeated = 1;
		char *is_chinese = strstr(name, "arrier");
		if (is_chinese == NULL) is_chinese = strstr(name, "ARRIER");

		if (is_chinese) {
			lpr = lpr_chinese_characters;
			output_depth = 71;
			output_len = 88;
		} else {
			lpr = lpr_characters;
			output_depth = 37;
			output_len = 106;
		}
		int unique[output_len];
		ctc_greedy_decode(unique, output, output_len, output_depth, merge_repeated);

		int i = 0;
		while(unique[i] != -1) {
			printf("'%s'", lpr[unique[i]]);
			i++;
		}
		printf("\n");


	} else if (!strcmp(str, "YOLOV2") || !strcmp(str, "YOLOV3") || !strcmp(str, "YOLOV4") || !strcmp(str, "YOLOV5") || !strcmp(str, "SSDV2")) {
		char **class_names = NULL;
		int valid_boxes = 0;
		fix16_box boxes[1024];
		int max_boxes = 100;
		fix16_t thresh = F16(0.3);
		fix16_t iou = F16(0.4);

		char *is_tiny = strstr(name, "iny");
		if (is_tiny == NULL) is_tiny = strstr(name, "INY");

		if(!strcmp(str, "YOLOV2")){ //tiny yolo v2
			int output_length = (int)(model_get_output_length(model, 0));
			int num_outputs = 1;
			fix16_t *outputs[] = {(fix16_t*)(uintptr_t)io_buffers[1]};

			if (output_length == 125*13*13) { // yolo v2 voc
				class_names = voc_classes;
				fix16_t tiny_anchors[] ={F16(1.08),F16(1.19),F16(3.42),F16(4.41),F16(6.63),F16(11.38),F16(9.42),F16(5.11),F16(16.620001),F16(10.52)};
				fix16_t anchors[] = {F16(1.3221),F16(1.73145),F16(3.19275),F16(4.00944),F16(5.05587),F16(8.09892),F16(9.47112),F16(4.84053),F16(11.2364),F16(10.0071)};
				yolo_info_t cfg_0 = {
					.version = 2,
					.input_dims = {3, 416, 416},
					.output_dims = {125, 13, 13},
					.coords = 4,
					.classes = 20,
					.num = 5,
					.anchors_length = 10,
					.anchors = is_tiny ? tiny_anchors : anchors,
				};
				yolo_info_t cfg[] = {cfg_0};
				valid_boxes = post_process_yolo(outputs, num_outputs, cfg, thresh, iou, boxes, max_boxes);
			} else if (output_length ==  425*13*13 || output_length == 425*19*19){ // yolo v2 coco
				int w, h, i;
				if (output_length == 425*13*13) {
					i = 416; h = 13; w = 13;
				} else {
					i = 608; h = 19; w = 19;
				}
				class_names = coco_classes;
				fix16_t anchors[] ={F16(0.57273),F16(0.677385),F16(1.87446),F16(2.06253),F16(3.33843),F16(5.47434),F16(7.88282),F16(3.52778),F16(9.77052),F16(9.16828)};
				yolo_info_t cfg_0 = {
					.version = 2,
					.input_dims = {3, i, i},
					.output_dims = {425, h, w},
					.coords = 4,
					.classes = 80,
					.num = 5,
					.anchors_length = 10,
					.anchors = anchors,
				};
				yolo_info_t cfg[] = {cfg_0};
				valid_boxes = post_process_yolo(outputs, num_outputs, cfg, thresh, iou, boxes, max_boxes);
			} else {
				printf("Please modify YOLOV2 post-processing config for your model\n");
				return -1;
			}


		} else if (!strcmp(str, "YOLOV3") || !strcmp(str, "YOLOV4")){ //tiny yolo v3/v4 COCO
			class_names = coco_classes;
			if (is_tiny) {
				int num_outputs = 2;
				fix16_t *outputs[2];
				int output_sizes[2] = {255*13*13, 255*26*26};
				for (int o = 0; o < num_outputs; o++) {
				  for (int i = 0; i < num_outputs; i++) {
				    if (model_get_output_length(model,i) == output_sizes[o]) {
					outputs[o] = (fix16_t*)(uintptr_t)io_buffers[i+1];
				    }
				  }
				}
				fix16_t tiny_anchors[] = {F16(10),F16(14),F16(23),F16(27),F16(37),F16(58),F16(81),F16(82),F16(135),F16(169),F16(344),F16(319)}; // 2*num
				int mask_0[] = {3,4,5};
				int mask_1[] = {1,2,3};

				yolo_info_t cfg_0 = {
					.version = 3,
					.input_dims = {3, 416, 416},
					.output_dims = {255, 13, 13},
					.coords = 4,
					.classes = 80,
					.num = 6,
					.anchors_length = 12,
					.anchors = tiny_anchors,
					.mask_length = 3,
					.mask = mask_0,
				};

				yolo_info_t cfg_1 = {
					.version = 3,
					.input_dims = {3, 416, 416},
					.output_dims = {255, 26, 26},
					.coords = 4,
					.classes = 80,
					.num = 6,
					.anchors_length = 12,
					.anchors = tiny_anchors,
					.mask_length = 3,
					.mask = mask_1,
				};

				yolo_info_t cfg[] = {cfg_0, cfg_1};

				valid_boxes = post_process_yolo(outputs, num_outputs, cfg, thresh, iou, boxes, max_boxes);
			} else {
				int num_outputs = 3;
				int output_sizes[3] = {255*19*19, 255*38*38, 255*76*76};
				int i = 608, o0 = 19, o1 = 38, o2 = 76;
				int is_416 =  model_get_input_length(model,0) == 3*416*416;
				if (is_416) {
					output_sizes[0] = 255*13*13;
					output_sizes[1] = 255*26*26;
					output_sizes[2] = 255*52*52;
					i = 416;
					o0 = 13;
					o1 = 26;
					o2 = 52;
				}
				fix16_t *outputs[3];
				for (int o = 0; o < num_outputs; o++) {
				  for (int i = 0; i < num_outputs; i++) {
				    if (model_get_output_length(model,i) == output_sizes[o]) {
					outputs[o] = (fix16_t*)(uintptr_t)io_buffers[i+1];
				    }
				  }
				}

				fix16_t anchors[] = {F16(10),F16(13),F16(16),F16(30),F16(33),F16(23),F16(30),F16(61),F16(62),F16(45),F16(59),F16(119),F16(116),F16(90),F16(156),F16(198),F16(373),F16(326)};
				int mask_0[] = {6,7,8};
				int mask_1[] = {3,4,5};
				int mask_2[] = {0,1,2};


				yolo_info_t cfg_0 = {
					.version = 3,
					.input_dims = {3, i, i},
					.output_dims = {255, o0, o0},
					.coords = 4,
					.classes = 80,
					.num = 9,
					.anchors_length = 18,
					.anchors = anchors,
					.mask_length = 3,
					.mask = mask_0,
				};

				yolo_info_t cfg_1 = {
					.version = 3,
					.input_dims = {3, i, i},
					.output_dims = {255, o1, o1},
					.coords = 4,
					.classes = 80,
					.num = 9,
					.anchors_length = 18,
					.anchors = anchors,
					.mask_length = 3,
					.mask = mask_1,
				};
				yolo_info_t cfg_2 = {
					.version = 3,
					.input_dims = {3, i, i},
					.output_dims = {255, o2, o2},
					.coords = 4,
					.classes = 80,
					.num = 9,
					.anchors_length = 18,
					.anchors = anchors,
					.mask_length = 3,
					.mask = mask_2,
				};

				yolo_info_t cfg[] = {cfg_0, cfg_1, cfg_2};

				valid_boxes = post_process_yolo(outputs, num_outputs, cfg, thresh, iou, boxes, max_boxes);
			}
		} else if (!strcmp(str, "YOLOV5")){ //ultralytics
			class_names = coco_classes;
			int num_outputs = 3;

			fix16_t *outputs[3];
			int output_sizes[3] = {255*13*13, 255*26*26, 255*52*52};
			for (int o = 0; o < num_outputs; o++) {
			  for (int i = 0; i < num_outputs; i++) {
			    if (model_get_output_length(model,i) == output_sizes[o]) {
				outputs[o] = (fix16_t*)(uintptr_t)io_buffers[i+1];
			    }
			  }
			}

			fix16_t anchors[] = {F16(10),F16(13),F16(16),F16(30),F16(33),F16(23),F16(30),F16(61),F16(62),F16(45),F16(59),F16(119),F16(116),F16(90),F16(156),F16(198),F16(373),F16(326)};
			int mask_0[] = {6,7,8};
			int mask_1[] = {3,4,5};
			int mask_2[] = {0,1,2};

			yolo_info_t cfg_0 = {
				.version = 5,
				.input_dims = {3, 416, 416},
				.output_dims = {255, 13, 13},
				.coords = 4,
				.classes = 80,
				.num = 9,
				.anchors_length = 18,
				.anchors = anchors,
				.mask_length = 3,
				.mask = mask_0,
			};

			yolo_info_t cfg_1 = {
				.version = 5,
				.input_dims = {3, 416, 416},
				.output_dims = {255, 26, 26},
				.coords = 4,
				.classes = 80,
				.num = 9,
				.anchors_length = 18,
				.anchors = anchors,
				.mask_length = 3,
				.mask = mask_1,
			};
			yolo_info_t cfg_2 = {
				.version = 5,
				.input_dims = {3, 416, 416},
				.output_dims = {255, 52, 52},
				.coords = 4,
				.classes = 80,
				.num = 9,
				.anchors_length = 18,
				.anchors = anchors,
				.mask_length = 3,
				.mask = mask_2,
			};

			yolo_info_t cfg[] = {cfg_0, cfg_1, cfg_2};

			valid_boxes = post_process_yolo(outputs, num_outputs, cfg, thresh, iou, boxes, max_boxes);

		}
		else if (!strcmp(str, "SSDV2")){
			fix16_t* output_buffers[12];
			char *is_vehicle = strstr(name, "ehicle");
			if (is_vehicle == NULL) is_vehicle = strstr(name, "EHICLE");

			fix16_t confidence_threshold=F16(0.5);
			fix16_t nms_threshold=F16(0.4);
			if (is_vehicle) {
				for(int o=0;o<6;++o){
					output_buffers[2*o]=(fix16_t*)(uintptr_t)io_buffers[1+(6-1-o)*2];
					output_buffers[2*o+1]=(fix16_t*)(uintptr_t)io_buffers[1+(6-1-o)*2+1];
				}
				valid_boxes = post_process_vehicles(boxes,max_boxes,output_buffers,3,confidence_threshold,nms_threshold);
				class_names = vehicle_classes;
			} else {
				for(int o=0;o<12;++o){
					output_buffers[o]=(fix16_t*)(uintptr_t)io_buffers[1+(12-1-o)];
				}
				valid_boxes = post_process_ssdv2(boxes,max_boxes,output_buffers,91,confidence_threshold,nms_threshold);
				class_names = coco91_classes;
			}
		}

		char class_str[50];
		for(int i=0;i<valid_boxes;++i){
			if(boxes[i].confidence == 0){
				continue;
			}

			if (class_names) { //class_names must be set, or prints the class id
				boxes[i].class_name = class_names[boxes[i].class_id];
				sprintf(class_str, "%s", boxes[i].class_name);
			} else {
				sprintf(class_str, "%d", boxes[i].class_id);
			}

			printf("%s\t%.2f\t(%d, %d, %d, %d)\n",
					class_str,
					fix16_to_float(boxes[i].confidence),
					boxes[i].xmin,boxes[i].xmax,
					boxes[i].ymin,boxes[i].ymax);
		}
	} else {
		printf("Unknown post processing type %s, skipping post process\n", str);
		return -1;
	}
	return 0;
}

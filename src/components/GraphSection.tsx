
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface GraphSectionProps {
  title: string;
  imageSrc: string;
  imageAlt: string;
}

const GraphSection = ({ title, imageSrc, imageAlt }: GraphSectionProps) => {
  return (
    <Card className="mt-6 shadow-sm border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800">
      <CardHeader className="bg-slate-50 dark:bg-slate-700/50">
        <CardTitle className="text-lg text-slate-800 dark:text-slate-200">
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent className="p-6">
        <div className="flex justify-center">
          <img 
            src={imageSrc}
            alt={imageAlt}
            className="max-w-full h-auto rounded-lg shadow-sm border border-slate-200 dark:border-slate-600"
            onError={(e) => {
              const target = e.target as HTMLImageElement;
              target.src = 'https://placehold.co/600x400/8b5cf6/ffffff?text=Graph+Loading...';
            }}
          />
        </div>
      </CardContent>
    </Card>
  );
};

export default GraphSection;
